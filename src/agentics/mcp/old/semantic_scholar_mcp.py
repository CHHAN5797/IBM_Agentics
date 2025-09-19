# semantic_scholar_mcp.py
# -------------------------------------------------------------
# Semantic Scholar MCP server (+ HTTP wrapper)
# Tools:
#   - search_papers, get_paper, get_citations, get_pdf_text
#   - research_dao_governance (builds a prompt; optionally calls an LLM)
# HTTP:
#   - GET /health
#   - GET /research_dao_governance
#
# Install:
#   python -m pip install -U "mcp>=1.2.0" "fastmcp>=0.4.3" httpx pypdf fastapi uvicorn python-dotenv
#
# Run (repo root):
#   uvicorn agentics.mcp.semantic_scholar_mcp:app --reload --port 8010 --app-dir src
#
# Test:
#   curl http://127.0.0.1:8010/health
#   curl "http://127.0.0.1:8010/research_dao_governance?prompt_mode=build&limit_recent=8"
#   # or override DOI explicitly:
#   curl "http://127.0.0.1:8010/research_dao_governance?prompt_mode=build&limit_recent=8&focal_doi=10.2139/ssrn.4367209"
# -------------------------------------------------------------

from __future__ import annotations

import io
import os
import re
import time
import json
import traceback
from typing import Any, Dict, List, Optional, Literal, TypedDict

# Load .env / .env.local if present
try:
    from dotenv import load_dotenv
    if os.path.exists(".env"):
        load_dotenv(".env", override=False)
    if os.path.exists(".env.local"):
        load_dotenv(".env.local", override=True)
except Exception:
    pass

import httpx
from httpx import HTTPStatusError
from mcp.server.fastmcp import FastMCP
from pypdf import PdfReader
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# ----------------------------
# Config
# ----------------------------
SEM_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")  # Optional but recommended
USER_AGENT = os.getenv("MCP_USER_AGENT", "dao-ai-semantic-scholar-mcp/1.0")

# Backoff for 429/5xx
MAX_RETRIES = int(os.getenv("MCP_MAX_RETRIES", "5"))
INITIAL_BACKOFF = float(os.getenv("MCP_INITIAL_BACKOFF", "0.8"))
MAX_BACKOFF = float(os.getenv("MCP_MAX_BACKOFF", "8.0"))
TIMEOUT_SECS = float(os.getenv("MCP_HTTP_TIMEOUT", "30.0"))

# Safe, token-friendly fields
DEFAULT_SEARCH_FIELDS = [
    "paperId", "title", "abstract", "year", "url",
    "authors.name", "authors.authorId",
    "openAccessPdf.url", "publicationDate", "citationCount",
]
DEFAULT_PAPER_FIELDS = [
    "paperId", "title", "abstract", "year", "url",
    "authors.name", "authors.authorId",
    "openAccessPdf.url", "publicationDate", "citationCount",
]
DEFAULT_CITATION_FIELDS = [
    "paperId", "title", "year",
    "authors.name", "authors.authorId",
    "openAccessPdf.url", "url",
]

# ----------------------------
# MCP App
# ----------------------------
mcp = FastMCP("semantic-scholar")

def _headers() -> Dict[str, str]:
    h = {"User-Agent": USER_AGENT}
    if API_KEY:
        h["x-api-key"] = API_KEY
    return h

def _client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT_SECS, headers=_headers(), follow_redirects=True)

def _norm_fields(fields: Optional[List[str]], default: List[str]) -> str:
    wanted = fields or default
    return ",".join(sorted(set(wanted)))

def _get_with_retry(client: httpx.Client, url: str, *, params: Optional[Dict[str, Any]] = None) -> httpx.Response:
    """HTTP GET with backoff for 429/5xx."""
    backoff = INITIAL_BACKOFF
    last_exc: Optional[Exception] = None
    for _ in range(1, MAX_RETRIES + 1):
        try:
            resp = client.get(url, params=params)
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                sleep_secs = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                time.sleep(min(sleep_secs, MAX_BACKOFF))
                backoff = min(backoff * 2.0, MAX_BACKOFF)
                continue
            resp.raise_for_status()
            return resp
        except httpx.HTTPError as e:
            last_exc = e
            time.sleep(backoff)
            backoff = min(backoff * 2.0, MAX_BACKOFF)
    if last_exc:
        raise last_exc
    raise RuntimeError("HTTP request failed with unknown error")

def _path_id(paper_id: str) -> str:
    """Do not URL-encode DOI slashes; S2 expects raw 'DOI:10.xxxx/yyy' in the path."""
    return paper_id.strip()

# ----------------------------
# DOI / metadata fallbacks
# ----------------------------
def _resolve_paper_id_from_doi(doi: str) -> Optional[str]:
    """
    Resolve a DOI to an S2 paperId via /paper/search by matching externalIds.DOI exactly.
    Returns an S2 paperId or None.
    """
    q = doi.strip()
    fields = "paperId,title,year,url,externalIds"
    with _client() as c:
        # try raw DOI
        r = _get_with_retry(c, f"{SEM_SCHOLAR_BASE}/paper/search",
                            params={"query": q, "fields": fields, "limit": 10})
        j = r.json() or {}
        for it in (j.get("data") or []):
            ext = (it.get("externalIds") or {})
            if (ext.get("DOI") or "").lower() == q.lower():
                return it.get("paperId")
        # also try "DOI:..." query form
        r2 = _get_with_retry(c, f"{SEM_SCHOLAR_BASE}/paper/search",
                             params={"query": f"DOI:{q}", "fields": fields, "limit": 10})
        j2 = r2.json() or {}
        for it in (j2.get("data") or []):
            ext = (it.get("externalIds") or {})
            if (ext.get("DOI") or "").lower() == q.lower():
                return it.get("paperId")
    return None

def _crossref_meta_from_doi(doi: str) -> Optional[Dict[str, Any]]:
    """
    Last-resort metadata when S2 can't resolve the DOI.
    Crossref is public; returns title/year/authors/url, sometimes abstract and PDF link.
    """
    url = f"https://api.crossref.org/works/{doi.strip()}"
    try:
        with httpx.Client(timeout=TIMEOUT_SECS, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as c:
            r = c.get(url)
            r.raise_for_status()
            msg = (r.json() or {}).get("message") or {}
    except Exception:
        return None

    title = (msg.get("title") or [None])[0]
    # year from issued / created / published-*
    year = None
    for k in ("issued", "created", "published-print", "published-online"):
        dp = ((msg.get(k) or {}).get("date-parts") or [[None]])[0]
        if dp and dp[0]:
            year = dp[0]
            break

    authors = [{"name": " ".join(filter(None, [a.get("given"), a.get("family")]))}
               for a in (msg.get("author") or [])]

    pdf_url = None
    for link in (msg.get("link") or []):
        if (link.get("content-type") or "").lower() == "application/pdf":
            pdf_url = link.get("URL")
            break

    abstract = msg.get("abstract")
    if abstract:
        abstract = re.sub(r"<[^>]+>", " ", abstract)
        abstract = re.sub(r"\s+", " ", abstract).strip()

    return {
        "paperId": f"DOI:{doi}",
        "title": title,
        "abstract": abstract,
        "year": year,
        "url": (msg.get("URL") or f"https://doi.org/{doi}"),
        "authors": authors,
        "openAccessPdf": {"url": pdf_url} if pdf_url else None,
        "publicationDate": None,
        "citationCount": None,
    }

def _openalex_pdf_from_doi(doi: str) -> Optional[str]:
    """
    Try to find an OA PDF via OpenAlex (no key required).
    Returns a direct PDF URL or None.
    """
    api = f"https://api.openalex.org/works/https://doi.org/{doi.strip()}"
    try:
        with httpx.Client(timeout=TIMEOUT_SECS, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as c:
            r = c.get(api)
            r.raise_for_status()
            j = r.json() or {}
    except Exception:
        return None

    loc = j.get("best_oa_location") or {}
    pdf = loc.get("url_for_pdf") or loc.get("pdf_url")
    if pdf:
        return pdf
    pl = j.get("primary_location") or {}
    return pl.get("pdf_url") or None

# ----------------------------
# Core Tools
# ----------------------------
@mcp.tool()
def search_papers(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 10,
    fields: Optional[List[str]] = None,
    sort: Literal["relevance", "citationCount:desc", "publicationDate:desc", "publicationDate:asc"] = "relevance",
) -> Dict[str, Any]:
    """Search papers on Semantic Scholar."""
    f = _norm_fields(fields, DEFAULT_SEARCH_FIELDS)
    params: Dict[str, Any] = {"query": query, "limit": max(1, min(limit, 100)), "fields": f, "sort": sort}
    if year_from is not None and year_to is not None:
        params["year"] = f"{year_from}-{year_to}"
    elif year_from is not None:
        params["year"] = f"{year_from}-{year_from}"
    elif year_to is not None:
        params["year"] = f"{year_to}-{year_to}"
    with _client() as c:
        resp = _get_with_retry(c, f"{SEM_SCHOLAR_BASE}/paper/search", params=params)
        return resp.json()

@mcp.tool()
def get_paper(paper_id: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Get a single paper by S2 paperId / DOI / ArXiv / CorpusId (with DOI fallback)."""
    f = _norm_fields(fields, DEFAULT_PAPER_FIELDS)
    try:
        with _client() as c:
            resp = _get_with_retry(c, f"{SEM_SCHOLAR_BASE}/paper/{_path_id(paper_id)}", params={"fields": f})
            return resp.json()
    except HTTPStatusError:
        # If DOI failed, try: DOI -> search resolve -> S2 paperId; else Crossref meta
        if str(paper_id).startswith("DOI:"):
            doi = paper_id.split("DOI:", 1)[1]
            resolved = _resolve_paper_id_from_doi(doi)
            if resolved:
                with _client() as c:
                    resp = _get_with_retry(c, f"{SEM_SCHOLAR_BASE}/paper/{resolved}", params={"fields": f})
                    return resp.json()
            cr = _crossref_meta_from_doi(doi)
            if cr:
                return cr
        raise

@mcp.tool()
def get_citations(
    paper_id: str,
    direction: Literal["references", "citations"] = "references",
    limit: int = 50,
    fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Expand the citation graph."""
    f = _norm_fields(fields, DEFAULT_CITATION_FIELDS)
    params = {"fields": f, "limit": max(1, min(limit, 100))}
    with _client() as c:
        resp = _get_with_retry(c, f"{SEM_SCHOLAR_BASE}/paper/{_path_id(paper_id)}/{direction}", params=params)
        raw = resp.json()
        items = raw.get("data") or raw.get("items") or []
        normalized = [(it.get("paper") if isinstance(it, dict) and "paper" in it else it) for it in items]
        return {"data": normalized, "total": raw.get("total")}

@mcp.tool()
def get_pdf_text(
    paper_id: str,
    max_pages: int = 12,
    prefer_cached_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Extract first N pages of an open-access PDF; fail safely when missing.
    Returns: paperId, title, year, pages_extracted, text, source_pdf, note
    """
    fields = ["title", "year", "openAccessPdf.url"]
    try:
        paper = get_paper(paper_id, fields=fields) if prefer_cached_metadata else {**get_paper(paper_id)}
    except Exception as e:
        return {
            "paperId": paper_id, "title": None, "year": None,
            "pages_extracted": 0, "text": None,
            "note": f"get_paper failed: {e!r}"
        }

    open_pdf = (paper.get("openAccessPdf") or {}).get("url")
    if not open_pdf:
        # If DOI, try OpenAlex OA PDF
        if str(paper_id).startswith("DOI:"):
            doi = paper_id.split("DOI:", 1)[1]
            oa_pdf = _openalex_pdf_from_doi(doi)
            if oa_pdf:
                open_pdf = oa_pdf

    if not open_pdf:
        return {
            "paperId": paper.get("paperId", paper_id),
            "title": paper.get("title"),
            "year": paper.get("year"),
            "pages_extracted": 0,
            "text": None,
            "note": "No openAccessPdf.url available",
        }

    try:
        with _client() as c:
            resp = _get_with_retry(c, open_pdf)
            reader = PdfReader(io.BytesIO(resp.content))
            n_pages = min(len(reader.pages), max(0, max_pages))
            chunks: List[str] = []
            for i in range(n_pages):
                try:
                    page_text = reader.pages[i].extract_text() or ""
                except Exception:
                    page_text = ""
                page_text = re.sub(r"\s+", " ", page_text).strip()
                if page_text:
                    chunks.append(page_text)
        return {
            "paperId": paper.get("paperId", paper_id),
            "title": paper.get("title"),
            "year": paper.get("year"),
            "pages_extracted": n_pages,
            "text": "\n\n".join(chunks) if chunks else None,
            "source_pdf": open_pdf,
        }
    except Exception as e:
        return {
            "paperId": paper.get("paperId", paper_id),
            "title": paper.get("title"),
            "year": paper.get("year"),
            "pages_extracted": 0,
            "text": None,
            "source_pdf": open_pdf,
            "note": f"pdf fetch/extract failed: {e!r}",
        }

# ----------------------------
# Helpers for robust focal resolution
# ----------------------------
def _normalize_doi(raw: str) -> str:
    """Normalize a DOI string (lowercase, strip spaces)."""
    return (raw or "").strip().lower()

def _title_from_semantic_url(url: str) -> Optional[str]:
    """Extract a human title from a Semantic Scholar paper URL slug."""
    try:
        slug = url.split("/paper/")[1].rsplit("/", 1)[0]
        title = " ".join(slug.replace("-", " ").split())
        return title
    except Exception:
        return None

def _match_paper_id_by_title(title: str) -> Optional[str]:
    """Use /paper/search/match to resolve a canonical paperId from a title string."""
    with _client() as c:
        r = _get_with_retry(c, f"{SEM_SCHOLAR_BASE}/paper/search/match",
                            params={"query": title, "fields": "paperId,title,url,year"})
        j = r.json() or {}
        if isinstance(j, dict):
            if j.get("data"):
                return j["data"][0].get("paperId")
            return j.get("paperId")
        return None

def _s2_id_from_url(url: str) -> Optional[str]:
    """Extract trailing hex-ish id from S2 URL (best-effort)."""
    m = re.search(r"/([0-9a-f]{10,40})(?:[#?/].*)?$", (url or "").strip(), re.I)
    return m.group(1) if m else None

# ----------------------------
# Research prompt tool (DAO Governance -> Snapshot metrics mapping)
# ----------------------------
class ResearchOut(TypedDict, total=False):
    topic: str
    focal_paper_id: Optional[str]
    focal_meta: Dict[str, Any]
    recent_hits: List[Dict[str, Any]]
    prompt: str
    llm_response: Optional[str]
    llm_json: Optional[Dict[str, Any]]
    note: Optional[str]

@mcp.tool()
def research_dao_governance(
    topic: str = "Decentralized Governance; DAO; on-chain voting; delegation; quadratic voting",
    # DOI-first (unique). Default to Appel & Grennan SSRN DOI:
    focal_doi: Optional[str] = "10.2139/ssrn.4367209",
    focal_s2_url: str = "https://www.semanticscholar.org/paper/Decentralized-Governance-and-Digital-Asset-Prices-Appel-Grennan/b7f1ecbb6bcfafc52f998ff3077ab64ad3951358",
    focal_title: Optional[str] = None,
    year_from: int = 2021,
    year_to: int = 2025,
    limit_recent: int = 12,
    max_pages_pdf: int = 8,
    prompt_mode: str = "build"  # "build" | "call"
) -> ResearchOut:
    """
    Build (or call) a prompt that:
      1) resolves the focal paper robustly (DOI -> S2 paperId; then title; then URL),
      2) pulls recent DAO governance papers around efficient decision-making,
      3) requests mapping to Snapshot MCP metrics.
    """
    notes: List[str] = []

    # 1) Resolve focal paperId: DOI > title match > URL id > URL as-is
    focal_id: Optional[str] = None
    if focal_doi:
        try:
            normalized = _normalize_doi(focal_doi)
            focal_id = f"DOI:{normalized}"
        except Exception as e:
            notes.append(f"[normalize doi error] {e!r}")

    if not focal_id:
        fixed_title = focal_title or _title_from_semantic_url(focal_s2_url)
        if fixed_title:
            try:
                focal_id = _match_paper_id_by_title(fixed_title)
            except Exception as e:
                notes.append(f"[match by title error] {e!r}")

    if not focal_id:
        maybe_hex = _s2_id_from_url(focal_s2_url)
        focal_id = maybe_hex if maybe_hex else focal_s2_url

    # Fetch focal meta (safe fields only)
    try:
        focal_meta = get_paper(focal_id, fields=[
            "paperId","title","abstract","year","url",
            "authors.name","openAccessPdf.url","publicationDate","citationCount"
        ])
    except Exception as e:
        focal_meta = {"paperId": focal_id, "title": None, "abstract": None, "url": focal_s2_url}
        notes.append(f"[focal get_paper error] {e!r}")

    # Try first pages text; fallback to abstract if needed
    focal_chunk = focal_meta.get("abstract") or "" if isinstance(focal_meta, dict) else ""
    try:
        fx = get_pdf_text(focal_id, max_pages=max_pages_pdf)
        if fx and fx.get("text"):
            focal_chunk = fx["text"]
        elif fx and fx.get("note"):
            notes.append(f"[focal get_pdf_text note] {fx.get('note')}")
    except Exception as e:
        notes.append(f"[focal get_pdf_text error] {e!r}")

    # 2) Recent hits (safe fields only)
    q = "DAO governance OR decentralized autonomous organization AND (governance OR voting OR delegation OR quadratic OR efficiency OR decision making)"
    try:
        hits = search_papers(
            query=f"{topic} {q}",
            year_from=year_from, year_to=year_to,
            limit=limit_recent, sort="publicationDate:desc",
            fields=[
                "paperId","title","abstract","year","url",
                "authors.name","openAccessPdf.url","publicationDate","citationCount"
            ],
        )
        items = hits.get("data") or hits.get("papers") or hits.get("items") or []
    except Exception as e:
        items = []
        notes.append(f"[search_papers error] {e!r}")

    # Build small snippets for token budget
    recent_summaries: List[Dict[str, Any]] = []
    for it in items[:limit_recent]:
        meta = {
            "paperId": it.get("paperId"),
            "title": it.get("title"),
            "year": it.get("year"),
            "url": it.get("url"),
            "citationCount": it.get("citationCount"),
            "authors": [a.get("name") for a in (it.get("authors") or [])],
            "openAccessPdf": (it.get("openAccessPdf") or {}).get("url"),
        }
        short = it.get("abstract") or ""
        if meta["openAccessPdf"]:
            try:
                txt = get_pdf_text(meta["paperId"], max_pages=4)
                if txt and txt.get("text"):
                    short = txt["text"][:4000]
                elif txt and txt.get("note"):
                    notes.append(f"[recent get_pdf_text {meta['paperId']}] {txt.get('note')}")
            except Exception as e:
                notes.append(f"[recent get_pdf_text {meta['paperId']}] {e!r}")
        meta["snippet"] = (short or "")[:2000]
        recent_summaries.append(meta)

    # 3) Prompt that maps to Snapshot metrics
    guidance = """
You are an expert in decentralized governance (DAOs). Read the focal paper and the recent papers list.
Goal: extract concrete procedures for more efficient decision making and map them to metrics we already compute from Snapshot (voting timeline & forum).

Snapshot metrics (already computed upstream, DO NOT recompute):
- Early lead hits & ratios by quartile (Q1~Q4), cumulative (count-based)
- Stability: penalize late leader flips after 75% of window
- VP by quartile; voting margin by quartile; overall margin
- Unique voters, total voting power (VP)

Tasks:
1) From papers, list procedures & institutional features that improve decision efficiency (e.g., delegation rules, quorum design, vote timing windows, quadratic/conviction voting, proposal templating, off-chain->on-chain handoffs).
2) For each procedure, define measurable indicators we can observe from Snapshot data (lead hits/ratios per quartile, turnout profile shapes, whale concentration, participation elasticity, time-to-first-lead, late-flip rate, abstain usage, etc.).
3) Produce a mapping table: {procedure -> expected direction on each Snapshot metric}, with short rationale grounded in citations.
4) Output a compact JSON with fields:
   {
     "procedures": [{"name":"...","paperId":"...","evidence":"<1-2 lines>"}],
     "indicators": [{"name":"...","definition":"...","source":"snapshot","unit":"..."}],
     "mapping": [{"procedure":"...","metric":"...","expected_effect":"increase|decrease|U-shape|none","evidence":"..."}],
     "reading_list": [{"paperId":"...","title":"...","year":2024,"url":"..."}]
   }
Use only the provided text/snippets. If a field is unknown, write null.
""".strip()

    prompt = (
        f"FOCAL PAPER:\n"
        f"{json.dumps({'meta': focal_meta, 'snippet': (focal_chunk or '')[:5000]}, ensure_ascii=False, indent=2)}\n\n"
        f"RECENT PAPERS (top {len(recent_summaries)}):\n"
        f"{json.dumps(recent_summaries, ensure_ascii=False, indent=2)}\n\n"
        f"{guidance}\n"
    )

    out: ResearchOut = {
        "topic": topic,
        "focal_paper_id": focal_id,
        "focal_meta": focal_meta,
        "recent_hits": recent_summaries,
        "prompt": prompt,
    }
    if notes:
        out["note"] = " | ".join(notes)

    if prompt_mode == "call":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            out["llm_response"] = "[DRY-RUN] No OPENAI_API_KEY set. Returning prompt only."
            return out
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            r = httpx.post(url, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            msg = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
            out["llm_response"] = msg
            try:
                out["llm_json"] = json.loads(msg)
            except Exception:
                out["llm_json"] = None
        except Exception as e:
            out["llm_response"] = f"[ERROR calling OpenAI] {e!r}"

    return out

# ----------------------------
# HTTP wrapper (curl-friendly; never raises 500 to client)
# ----------------------------
fastapi_app = FastAPI(title="Semantic Scholar MCP + HTTP", version="1.0")

@fastapi_app.get("/health")
def health():
    return {
        "ok": True,
        "keys_loaded": {
            "semantic_scholar": bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "openai_model": os.getenv("OPENAI_MODEL") or os.getenv("OPENAI_MODEL_ID") or None,
        }
    }

@fastapi_app.get("/research_dao_governance")
def research_dao_governance_http(
    topic: str = Query("Decentralized Governance; DAO; on-chain voting; delegation; quadratic voting"),
    focal_doi: Optional[str] = Query("10.2139/ssrn.4367209"),
    focal_s2_url: str = Query("https://www.semanticscholar.org/paper/Decentralized-Governance-and-Digital-Asset-Prices-Appel-Grennan/b7f1ecbb6bcfafc52f998ff3077ab64ad3951358"),
    focal_title: Optional[str] = Query(None),
    year_from: int = Query(2021),
    year_to: int = Query(2025),
    limit_recent: int = Query(12),
    max_pages_pdf: int = Query(8),
    prompt_mode: str = Query("build")
):
    try:
        return research_dao_governance(
            topic=topic,
            focal_doi=focal_doi,
            focal_s2_url=focal_s2_url,
            focal_title=focal_title,
            year_from=year_from,
            year_to=year_to,
            limit_recent=limit_recent,
            max_pages_pdf=max_pages_pdf,
            prompt_mode=prompt_mode
        )
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "error": f"{e.__class__.__name__}: {e}",
                "trace": traceback.format_exc(),
                "note": "Handled gracefully so you can see what failed."
            }
        )

# Mount the MCP protocol app at /mcp for IDE/LLM clients
try:
    fastapi_app.mount("/mcp", getattr(mcp, "app", None) or getattr(mcp, "asgi", None))
except Exception:
    pass

# uvicorn ASGI entrypoint
app = fastapi_app


## Semantics API fix (change focal paper to one with Open access PDF) or other papers without specifying focal_doi
## check if Open AI can summarize the paper