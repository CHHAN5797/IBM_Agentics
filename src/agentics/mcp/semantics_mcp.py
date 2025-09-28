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
    Returns a list of dicts; may contain None fields if S2 lacks metadata.
    """
    _sleep()
    url = f"{S2_BASE}/paper/search"
    params = {
        "query": query,
        "limit": k,
        "fields": "title,year,authors,url,abstract,externalIds,publicationTypes,venue"
    }
    r = session.get(url, headers=_headers(), params=params, timeout=TIMEOUT)
    if r.status_code != 200:
        return []
    data = r.json().get("data", []) or []
    return data


def s2_get(paper_id_or_doi: str) -> Dict[str, Any]:
    """
    Fetch one paper by S2 paperId or DOI:... handle missing meta gracefully.
    """
    _sleep()
    pid = paper_id_or_doi
    url = f"{S2_BASE}/paper/{requests.utils.quote(pid)}"
    params = {
        "fields": "title,year,authors,url,abstract,externalIds,publicationTypes,venue"
    }
    r = session.get(url, headers=_headers(), params=params, timeout=TIMEOUT)
    if r.status_code == 404:
        # Graceful fallback with empty metadata (caller can still proceed)
        return {
            "paperId": None, "title": None, "abstract": None, "year": None,
            "authors": [], "url": None, "externalIds": {}, "venue": None,
        }
    r.raise_for_status()
    return r.json()


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
    queries: Annotated[List[str], Field(
        description="List of search queries for academic papers",
        min_length=1,
        max_length=10
    )],
    per_query: Annotated[int, Field(
        description="Maximum number of papers to retrieve per query",
        ge=1,
        le=20
    )] = 5,
    enrich: Annotated[bool, Field(
        description="Whether to fetch detailed metadata for each paper"
    )] = True
) -> List[Dict[str, Any]]:
    """
    Search papers for a list of queries. If `enrich` is True, fetch each paper again by id/DOI.
    Returns a de-duplicated list of normalized papers.
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for q in queries:
        for hit in s2_search(q, k=max(1, min(per_query, 20))):
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
def plan_prompt(
    goal: Annotated[str, Field(
        description="Research goal or objective statement",
        min_length=1,
        max_length=1000
    )],
    y_vars: Annotated[Optional[List[str]], Field(
        description="List of dependent variables or outcomes to measure"
    )] = None,
    domains: Annotated[Optional[List[str]], Field(
        description="Research domains or fields of study"
    )] = None,
    queries: Annotated[Optional[List[str]], Field(
        description="Literature search queries for supporting research"
    )] = None,
    per_query: Annotated[int, Field(
        description="Number of papers to find per query",
        ge=1,
        le=20
    )] = 5
) -> Dict[str, Any]:
    """
    Build a JSON-return instruction prompt for LLM planning.
    This does NOT execute any other MCP; it only prepares prompt text.
    """
    queries = queries or []
    papers = find_papers(queries, per_query=per_query, enrich=True) if queries else []
    prompt = render_plan_prompt(goal=goal, y_vars=y_vars or [], domains=domains or [], candidate_papers=papers)
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
