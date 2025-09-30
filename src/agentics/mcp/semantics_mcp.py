
from __future__ import annotations
"""
Semantics-Only MCP (FastMCP, stdio)
- Focus: literature retrieval + metric blueprints + prompt generation + BibTeX helpers
- No orchestration of other MCPs (Snapshot/Forums/Timeline etc.). This module only prepares
  the *research plan prompts* and *interpretation prompts* + structured references.
"""

import os
import time
import json
import textwrap
from typing import List, Dict, Any, Optional, Annotated

import requests
from mcp.server.fastmcp import FastMCP
from pydantic import Field

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
UA = "semantics-only-mcp/0.2"
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()

TIMEOUT = int(os.getenv("SEMANTICS_HTTP_TIMEOUT", "30"))
BASE_SLEEP = float(os.getenv("SEMANTICS_BASE_SLEEP", "1"))

session = requests.Session()
session.headers.update({"User-Agent": UA})

# === NEW: rate limit, retries, cache =========================================
import threading
import random

SEMANTICS_MAX_RPM = int(os.getenv("SEMANTICS_MAX_RPM", "30"))  
SEMANTICS_MAX_RETRIES = int(os.getenv("SEMANTICS_MAX_RETRIES", "4"))
SEMANTICS_MAX_BACKOFF = float(os.getenv("SEMANTICS_MAX_BACKOFF", "16"))

_min_interval = 60.0 / max(1, SEMANTICS_MAX_RPM)
_last_call_ts = 0.0
_lock = threading.Lock()


_cache: Dict[str, Any] = {}

def _cache_key(url: str, params: Optional[Dict[str, Any]]) -> str:
    items = tuple(sorted((params or {}).items()))
    return f"{url}|{items}"

def _rate_limit_sleep():
    global _last_call_ts
    with _lock:
        now = time.time()
        wait = _min_interval - (now - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
            _last_call_ts = time.time()
        else:
            _last_call_ts = now

def _request_with_retries(method: str, url: str, *, params=None, timeout=None, headers=None):
    key = _cache_key(url, params)
    if method.upper() == "GET" and key in _cache:
        return _cache[key]

    attempt = 0
    backoff = BASE_SLEEP if BASE_SLEEP > 0 else 0.5

    while True:
        _rate_limit_sleep()
        try:
            resp = session.request(method=method, url=url, params=params, timeout=timeout, headers=headers or _headers())
        except requests.RequestException:
            attempt += 1
            if attempt > SEMANTICS_MAX_RETRIES:
                raise
            time.sleep(min(SEMANTICS_MAX_BACKOFF, backoff + random.uniform(0, backoff)))
            backoff *= 2
            continue

        if 200 <= resp.status_code < 300:
            if method.upper() == "GET":
                _cache[key] = resp
            return resp


        if resp.status_code in (429, 500, 502, 503, 504):
            attempt += 1
            if attempt > SEMANTICS_MAX_RETRIES:
                return resp  

            ra = resp.headers.get("Retry-After")
            if ra:
                try:
                    sleep_s = float(ra)
                except Exception:
                    sleep_s = min(SEMANTICS_MAX_BACKOFF, backoff)
            else:
                sleep_s = min(SEMANTICS_MAX_BACKOFF, backoff + random.uniform(0, backoff))
            time.sleep(sleep_s)
            backoff = min(SEMANTICS_MAX_BACKOFF, backoff * 2)
            continue

        return resp



def _sleep():
    time.sleep(BASE_SLEEP)


def _headers() -> Dict[str, str]:
    """Attach Semantic Scholar API key if available."""
    h = {"Accept": "application/json"}
    if S2_KEY:
        h["x-api-key"] = S2_KEY
    return h


# --------------------------------------------------------------------------------------
# Minimal S2 client (with graceful handling when title/abstract are missing)
# --------------------------------------------------------------------------------------
def s2_search(query: str, k: int = 8) -> List[Dict[str, Any]]:
    """
    Search Semantic Scholar for papers (title/year/authors/url/abstract/externalIds).
    Returns a list; may contain None fields.
    - 안전: rate limit, retries, 캐시, 429/5xx 대응
    """
    url = f"{S2_BASE}/paper/search"
    params = {
        "query": query,
        "limit": max(1, min(k, 20)),
        "fields": "title,year,authors,url,abstract,externalIds,publicationTypes,venue"
    }
    resp = _request_with_retries("GET", url, params=params, timeout=TIMEOUT, headers=_headers())

    if resp.status_code != 200:
        try:
            _ = resp.text  # noqa
        except Exception:
            pass
        return []

    data = resp.json().get("data", []) or []
    return data
def s2_get(paper_id_or_doi: str) -> Dict[str, Any]:
    """
    Fetch one paper by S2 paperId or DOI:... handle missing meta gracefully.
    - 안전: rate limit, retries, 캐시, 404 graceful
    """
    pid = paper_id_or_doi
    url = f"{S2_BASE}/paper/{requests.utils.quote(pid)}"
    params = {
        "fields": "title,year,authors,url,abstract,externalIds,publicationTypes,venue"
    }
    resp = _request_with_retries("GET", url, params=params, timeout=TIMEOUT, headers=_headers())

    if resp.status_code == 404:
        return {
            "paperId": None, "title": None, "abstract": None, "year": None,
            "authors": [], "url": None, "externalIds": {}, "venue": None,
        }

    if resp.status_code != 200:
        return {
            "paperId": None, "title": None, "abstract": None, "year": None,
            "authors": [], "url": None, "externalIds": {}, "venue": None,
        }

    return resp.json()


def _normalize_paper(js: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize S2 response to a stable, small schema for LLM prompts and logs.
    """
    ext = (js.get("externalIds") or {})
    return {
        "paperId": js.get("paperId"),
        "doi": ext.get("DOI"),
        "title": js.get("title"),
        "authors": [a.get("name") for a in (js.get("authors") or []) if isinstance(a, dict)],
        "year": js.get("year"),
        "url": js.get("url"),
        "venue": js.get("venue"),
        "abstract": js.get("abstract"),
    }


# --------------------------------------------------------------------------------------
# Prompt builders (LLM-facing). These return *strings* you can feed to your agent.
# --------------------------------------------------------------------------------------
def render_plan_prompt(
    goal: str,
    y_vars: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    candidate_papers: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Build a strict JSON instruction prompt that asks the LLM to return a ResearchPlan.
    The caller is responsible for *function calling* or JSON schema enforcement.
    """
    y_vars = y_vars or []
    domains = domains or []
    papers = candidate_papers or []

    papers_block = json.dumps(papers, ensure_ascii=False, indent=2)
    schema_hint = textwrap.dedent(
        """
        Return STRICT JSON with this schema:
        {
          "objectives": string[],
          "metrics": [{
            "name": string,
            "purpose": string,
            "formula": string,                     // math or pseudo-code
            "required_data": string[],             // data names only, no code
            "estimator": string|null,              // method, test, regression, etc.
            "references": [{
              "title": string|null,
              "doi": string|null,
              "year": number|null,
              "authors": string[],
              "url": string|null
            }]
          }],
          "data_plan": {
            "requests": [{
              "name": string,                      // stable name to be referenced later
              "source": "snapshot|forums|prices|defi|custom", // ONLY labels; no execution
              "needs": string[]                    // what fields are required (e.g., 'proposal_id')
            }]
          },
          "eval_plan": {
            "robustness": string[],
            "notes": string
          }
        }
        """
    ).strip()

    return textwrap.dedent(f"""
    You are an academic research planner. Given a project's goal, propose metrics and a data plan.

    GOAL:
    {goal}

    Y-VARIABLES OF INTEREST (may appear in objectives or eval_plan):
    {json.dumps(y_vars, ensure_ascii=False)}

    DOMAINS:
    {json.dumps(domains, ensure_ascii=False)}

    CANDIDATE PAPERS (metadata; may contain null fields):
    {papers_block}

    TASK:
    - Propose metrics connecting the goal and domains to measurable quantities.
    - For each metric, specify purpose, formula (text/LaTeX/pseudo), required_data, estimator.
    - Attach relevant references from CANDIDATE PAPERS when appropriate.
    - Propose a data_plan with *labels only* (no tool names, no code execution).
    - Propose eval_plan focusing on identification limits and robustness.

    {schema_hint}

    IMPORTANT:
    - Do NOT claim causality; observational limits must be explicit.
    - Do NOT include any code or external calls.
    - Keep names stable and succinct. Ensure JSON validity.
    """).strip()


def render_interpretation_prompt(
    plan: Dict[str, Any],
    computed_metrics: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Build an interpretation prompt. Caller passes computed metrics later (from other MCPs),
    but this module itself does not orchestrate or compute them.
    """
    return textwrap.dedent(f"""
    You are an econometrics tutor. Interpret the results of the metrics below in a careful,
    non-causal way. Summarize key insights, caveats, and robustness checks to consider.
    Return STRICT JSON:
    {{
      "overview": string,
      "interpretation": string,
      "limitations": string,
      "next_steps": string[]
    }}

    RESEARCH PLAN:
    {json.dumps(plan, ensure_ascii=False, indent=2)}

    (OPTIONAL) COMPUTED METRICS:
    {json.dumps(computed_metrics or [], ensure_ascii=False, indent=2)}

    IMPORTANT:
    - Emphasize observational nature and sample-period limits.
    - Flag data quality issues and missing variables that could bias conclusions.
    - Keep JSON compact and valid.
    """).strip()


def _to_bibtex(p: Dict[str, Any]) -> str:
    """Very lightweight BibTeX generator from normalized paper meta."""
    key = (p.get("doi") or p.get("paperId") or "ref").replace("/", "_") if (p.get("doi") or p.get("paperId")) else "ref"
    authors = " and ".join(p.get("authors") or [])
    year = p.get("year") or ""
    title = p.get("title") or ""
    venue = p.get("venue") or ""
    doi = p.get("doi") or ""
    url = p.get("url") or ""
    return textwrap.dedent(f"""\
    @misc{{{key},
      title = {{{title}}},
      author = {{{authors}}},
      year = {{{year}}},
      howpublished = {{{venue}}},
      doi = {{{doi}}},
      url = {{{url}}}
    }}""")


# --------------------------------------------------------------------------------------
# FastMCP tools (Semantics-only)
# --------------------------------------------------------------------------------------
mcp = FastMCP("SemanticsOnlyMCP")


@mcp.tool(
    name="health",
    title="Semantics Service Health Check",
    description="Check the health status of the Semantics MCP service. Verifies Semantic Scholar API key availability and service readiness.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
def health() -> Dict[str, Any]:
    """
    Basic health with a check on whether the Semantic Scholar key is present.
    The key is optional; this MCP still works with degraded metadata without it.
    """
    return {
        "ok": True,
        "service": "SemanticsOnlyMCP",
        "has_s2_key": bool(S2_KEY),
        "ua": UA
    }


@mcp.tool(
    name="find_papers",
    title="Search Academic Papers",
    description="Search for academic papers using multiple queries via Semantic Scholar API. Returns de-duplicated normalized paper metadata with optional enrichment. Use this for literature research and academic reference gathering.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
def find_papers(
    queries: List[str],
    per_query: int = 5,
    enrich: bool = False   
) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for q in queries:
        hits = s2_search(q, k=max(1, min(per_query, 20)))
        for hit in hits:
            pid = hit.get("paperId") or (hit.get("externalIds") or {}).get("DOI")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            meta = s2_get(pid) if enrich else hit
            out.append(_normalize_paper(meta))
    return out



@mcp.tool(
    name="plan_prompt",
    title="Generate Research Plan Prompt",
    description="Build a structured research plan prompt for LLM planning with literature support. Creates prompts for research planning with academic references and methodology guidance.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)

# === PATCH: plan_prompt 교체 ===
def plan_prompt(
    goal: Annotated[str, Field(min_length=1, max_length=1000)],
    y_vars: Annotated[Optional[List[str]], Field(description="List of dependent variables or outcomes to measure")] = None,
    domains: Annotated[Optional[List[str]], Field(description="Research domains or fields of study")] = None,
    queries: Annotated[Optional[List[str]], Field(description="Literature search queries for supporting research")] = None,
    per_query: Annotated[int, Field(ge=1, le=20, description="Number of papers to find per query")] = 5,
    use_literature: Annotated[bool, Field(description="Whether to actually run literature search")] = True,
    year_cap: Annotated[Optional[int], Field(description="Prefer papers with year <= year_cap; best-effort")] = None,
    enrich: Annotated[bool, Field(description="Enrich with s2_get only if truly needed")] = False,
) -> Dict[str, Any]:
    """
    Build a JSON-return instruction prompt for LLM planning.
    Literature search is optional (controlled by use_literature).
    """
    queries = (queries or [])[:2]   # guardrail: 최대 1–2개로 제한
    papers: List[Dict[str, Any]] = []

    if use_literature and queries:
        # 연도 캡이 있으면 연도 필터 우선, 아니면 기본 검색
        if year_cap is not None:
            # post-filter + normalize, 호출수 최소화
            for q in queries:
                # s2_search_year_capped은 이미 normalize해서 반환
                papers.extend(s2_search_year_capped(q, k=min(per_query, 4), year_cap=year_cap))
        else:
            papers = find_papers(queries=queries, per_query=min(per_query, 4), enrich=enrich)

    prompt = render_plan_prompt(
        goal=goal,
        y_vars=y_vars or [],
        domains=domains or [],
        candidate_papers=papers
    )
    return {
        "mode": "build",
        "papers_found": papers,
        "prompt": prompt
    }


@mcp.tool(
    name="interpretation_prompt",
    title="Generate Interpretation Prompt",
    description="Build interpretation prompts from completed research plans and computed metrics. Creates structured prompts for analyzing and interpreting research results with academic context.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
def interpretation_prompt(
    plan_json: Annotated[Dict[str, Any], Field(
        description="Completed research plan as JSON object"
    )],
    computed_metrics: Annotated[Optional[List[Dict[str, Any]]], Field(
        description="List of computed metrics and results from analysis"
    )] = None
) -> Dict[str, Any]:
    """
    Build an interpretation prompt from a completed plan and (optionally) externally-computed metrics.
    """
    prompt = render_interpretation_prompt(plan=plan_json, computed_metrics=computed_metrics or [])
    return {"prompt": prompt}


@mcp.tool(
    name="bibtex_from_dois",
    title="Generate BibTeX from DOIs",
    description="Generate formatted BibTeX citations from a list of DOI identifiers using Semantic Scholar metadata. Use this for academic citation management and bibliography generation.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
def bibtex_from_dois(
    dois: Annotated[List[str], Field(
        description="List of DOI identifiers for papers to cite",
        min_length=1,
        max_length=50
    )]
) -> str:
    """
    Generate a BibTeX block from a list of DOIs (best-effort).
    Missing metadata fields are tolerated.
    """
    items = []
    for d in dois:
        js = s2_get(f"DOI:{d}") if not d.upper().startswith("DOI:") else s2_get(d)
        items.append(_to_bibtex(_normalize_paper(js)))
    return "\n\n".join(items)


@mcp.tool(
    name="bibtex_from_papers",
    title="Generate BibTeX from Paper Objects",
    description="Generate formatted BibTeX citations from paper metadata objects. Use this to create bibliographies from already-retrieved paper data.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
def bibtex_from_papers(
    papers: Annotated[List[Dict[str, Any]], Field(
        description="List of paper metadata objects to convert to BibTeX",
        min_length=1,
        max_length=50
    )]
) -> str:
    """
    Generate a BibTeX block from already-normalized paper dicts (or close enough).
    """
    items = []
    for p in papers:
        items.append(_to_bibtex(_normalize_paper(p)))
    return "\n\n".join(items)


if __name__ == "__main__":
    # Run as stdio MCP server so Agentics/your agent can attach directly.
    mcp.run(transport="stdio")

def s2_search_year_capped(query: str, k: int, year_cap: int) -> List[Dict[str, Any]]:
    """
    Search then post-filter by year <= year_cap. Returns normalized list.
    """
    try:
        raw = s2_search(query, k=k)
    except Exception:
        raw = []
    out = []
    for p in raw or []:
        y = p.get("year")
        try:
            if y is None or int(y) <= int(year_cap):
                out.append(_normalize_paper(p))
        except Exception:
            out.append(_normalize_paper(p))
    return out[:k]

def build_bibliography(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Minimal structured references for logging/output.
    """
    refs = []
    for p in papers or []:
        refs.append({
            "title": p.get("title"),
            "year": p.get("year"),
            "authors": p.get("authors"),
            "venue": p.get("venue"),
            "doi": p.get("doi"),
            "url": p.get("url")
        })
    return refs
