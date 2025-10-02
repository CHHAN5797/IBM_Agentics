from __future__ import annotations

"""
Interactive Agentics — governance vote recommendation (ex‑post blind planning)
with deterministic Timeline MCP usage, LLM‑driven forum sentiment, and robust
price/TVL impact computation (including past proposals).

What this orchestrator guarantees now:
- Snapshot votes are fetched exhaustively (MCP → fallback GraphQL), then
  Timeline MCP computes metrics BEFORE the agent runs.
- Forum sentiment is done by the LLM (no hardcoded rules). We also print the
  number of posts analyzed and top‑3 quotes (positive/negative) to console.
- Token price impact is computed from a local CMC parquet for the current
  proposal **and** recent past proposals for context. Parquet path may be
  overridden via ENV CMC_OFFLINE_PARQUET. UCID is normalized.
- TVL impact via DeFiLlama MCP is more robust (link/slug/resolve flow), with
  diagnostics printed when unavailable.

Drop‑in replacement for your previous orchestrator.
Comments are in English for clarity.
"""

import asyncio
import csv
import json
import os
import random
import re
import time
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from crewai_tools import MCPServerAdapter
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agentics import Agentics as AG
from agentics.core.llm_connections import get_llm_provider
from mcp import StdioServerParameters

# --- CrewAI FilteredStream flush guard (unchanged) ---
try:
    import crewai.llm as _crewai_llm

    _orig_flush = _crewai_llm.FilteredStream.flush

    def _safe_flush(self):
        try:
            return _orig_flush(self)
        except ValueError:
            return None

    _crewai_llm.FilteredStream.flush = _safe_flush
except Exception:
    pass

# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------


def _to_dict(m):
    if hasattr(m, "model_dump"):
        return m.model_dump()
    if hasattr(m, "dict"):
        return m.dict()
    return m


def _to_json_str(m, indent=2):
    try:
        return json.dumps(_to_dict(m), indent=indent, ensure_ascii=False)
    except Exception:
        if hasattr(m, "json"):
            return m.json(indent=indent, ensure_ascii=False)
        if hasattr(m, "model_dump_json"):
            return m.model_dump_json(indent=indent)
        return json.dumps(m, indent=indent, ensure_ascii=False)


# ----------------------------------------------------------------------------
# Output schemas (extended with forum sentiment & post_count)
# ----------------------------------------------------------------------------
class Evidence(BaseModel):
    source_tool: str
    reference: Optional[str] = None
    quote: Optional[str] = None


class ActualVoteResult(BaseModel):
    winner_label: Optional[str] = None
    winner_index: Optional[int] = None
    scores: Optional[List[float]] = None
    scores_total: Optional[float] = None
    margin_abs: Optional[float] = None
    margin_pct: Optional[float] = None


class ForumSentiment(BaseModel):
    overall_polarity: Optional[float] = Field(None, description="-1..1")
    stance_share: Optional[Dict[str, float]] = Field(
        default=None, description="Fractional shares for {'for','against','neutral'}"
    )
    toxicity_flags: Optional[List[str]] = None
    key_topics: Optional[List[str]] = None
    top_positive_quotes: Optional[List[str]] = None
    top_negative_quotes: Optional[List[str]] = None
    summary: Optional[str] = None
    post_count: Optional[int] = None  # NEW: number of posts analyzed


class ProposalDecision(BaseModel):
    snapshot_url: str
    selected_choice_label: str
    selected_choice_index: Optional[int] = None
    confidence: float
    summary: str
    key_arguments_for: List[str] = Field(default_factory=list)
    key_arguments_against: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    available_choices: Optional[List[str]] = None
    event_start_utc: Optional[str] = None
    event_end_utc: Optional[str] = None
    event_time_utc: Optional[str] = None
    address_of_governance_token: Optional[str] = None
    token_price_impact_pct: Optional[float] = None
    tvl_impact_pct: Optional[float] = None
    actual_vote_result: Optional[ActualVoteResult] = None
    simulation_reason: Optional[str] = None
    references: Optional[List[Dict[str, Any]]] = None
    # New fields
    forum_sentiment: Optional[ForumSentiment] = None
    timeline_metrics: Optional[Dict[str, Any]] = None


# Planning schemas
class ToolCall(BaseModel):
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    why: str


class ToolPlan(BaseModel):
    objective: str
    calls: List[ToolCall]
    notes: Optional[str] = None


# ----------------------------------------------------------------------------
# MCP wiring
# ----------------------------------------------------------------------------


def _server_params(
    script_path: Path, src_dir: Path, extra_env: Optional[Dict[str, str]] = None
) -> StdioServerParameters:
    env = {"PYTHONPATH": str(src_dir), **os.environ}
    if extra_env:
        env.update(extra_env)
    return StdioServerParameters(command="python3", args=[str(script_path)], env=env)


def _load_mcp_tools(project_root: Path, stack: ExitStack) -> List:
    """Start available MCP servers and concatenate their tool adapters.
    Skips servers that are missing locally, so smoke tests still run.
    """
    src_dir = project_root / "src"
    cmc_parquet = _resolve_parquet_path(project_root)
    cmc_env = (
        {"CMC_OFFLINE_PARQUET": str(cmc_parquet)} if cmc_parquet.exists() else None
    )

    server_specs = [
        ("snapshot", project_root / "src/agentics/mcp/snapshot_api.py", None),
        ("timeline", project_root / "src/agentics/mcp/timeline_mcp.py", None),
        ("forums", project_root / "src/agentics/mcp/forums_mcp.py", None),
        ("defillama", project_root / "src/agentics/mcp/defillama_mcp.py", None),
        ("holders", project_root / "src/agentics/mcp/holders_activity_mcp.py", None),
        (
            "onchain",
            project_root / "src/agentics/mcp/onchain_activity_mcp_bq_cmc.py",
            None,
        ),
        ("cmc", project_root / "src/agentics/mcp/cmc_offline_mcp.py", cmc_env),
        ("semantics", project_root / "src/agentics/mcp/semantics_mcp.py", None),
        ("registry", project_root / "src/agentics/mcp/registry_mcp.py", None),
        ("govnews", project_root / "src/agentics/mcp/govnews_mcp.py", None),
    ]

    combined = None
    for name, script_path, extra_env in server_specs:
        if not script_path.exists():
            print(f"[skip] {name}: missing server at {script_path}")
            continue
        adapter = stack.enter_context(
            MCPServerAdapter(_server_params(script_path, src_dir, extra_env))
        )
        print(f"Available {name} tools: {[tool.name for tool in adapter]}")
        combined = adapter if combined is None else combined + adapter
    if combined is None:
        raise RuntimeError(
            "No MCP servers could be started. Check dependencies and configuration."
        )
    return combined


def _blind_toolset(tools) -> List:
    banned = {"get_proposal_result_by_id"}
    return [t for t in tools if getattr(t, "name", None) not in banned]


# ----------------------------------------------------------------------------
# Tool invocation shim — robust across call styles
# ----------------------------------------------------------------------------


def _invoke_tool(tools, name: str, **kwargs):
    tool = next((t for t in tools if getattr(t, "name", None) == name), None)
    if tool is None:
        print(
            f"[invoke] Tool not found: {name}. Available: {[getattr(t,'name',None) for t in tools]}"
        )
        return None
    payload = dict(kwargs)
    # Try common invocation surfaces
    for attr in ("call", "run"):
        if hasattr(tool, attr):
            try:
                return getattr(tool, attr)(**payload)
            except Exception as e:
                print(f"[invoke] {name}.{attr}(**) failed: {e}")
    # Fallback variations (dict payload)
    for attr in ("call", "run"):
        if hasattr(tool, attr):
            try:
                return getattr(tool, attr)(payload)
            except Exception as e:
                print(f"[invoke] {name}.{attr}(dict) failed: {e}")
    if callable(tool):
        try:
            return tool(**kwargs)
        except Exception as e:
            print(f"[invoke] {name}(**kwargs) failed: {e}")
    print(f"[invoke] {name}: no supported invocation method.")
    return None


def _invoke_tool_try_names_and_params(
    tools, names: List[str], param_variants: List[Dict[str, Any]]
) -> Optional[dict]:
    for nm in names:
        for params in param_variants:
            res = _invoke_tool(tools, nm, **params)
            if res is not None:
                return res
    print(
        f"[invoke] Tried names={names} with {len(param_variants)} param variants, all failed."
    )
    print("[invoke] Available tools:", [getattr(t, "name", None) for t in tools])
    return None


# ----------------------------------------------------------------------------
# Snapshot GraphQL (orchestrator‑side helpers)
# ----------------------------------------------------------------------------

SNAPSHOT_API = os.getenv("SNAPSHOT_API", "https://hub.snapshot.org/graphql")
_HTTP = requests.Session()
_HTTP.headers.update({"User-Agent": "agentics-orchestrator/1.4"})

PROPOSAL_RESULT_Q = """
query($id: String!) {
  proposal(id: $id) {
    id
    choices
    scores
    scores_total
    state
    start
    end
  }
}
"""

PROPOSAL_BY_ID_Q = """
query ($id: String!) {
  proposal(id: $id) {
    id
    title
    body
    author
    choices
    discussion
    start
    end
    state
    space { id name strategies { name network params } }
  }
}
"""

VOTES_BY_PROPOSAL_Q = """
query($proposal: String!, $first: Int!, $skip: Int!) {
  votes(first:$first, skip:$skip, where:{ proposal:$proposal }, orderBy:"created", orderDirection:asc){
    id voter created choice vp reason
  }
}
"""

PROPOSALS_BY_SPACE_Q = """
query($space: String!, $first: Int!, $skip: Int!) {
  proposals(first:$first, skip:$skip, where:{ space_in: [$space] }, orderBy:"created", orderDirection:desc){
    id title body author start end state choices
  }
}
"""


def _gql(query: str, variables: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    retries = 0
    backoff = 1.7
    while True:
        try:
            r = _HTTP.post(
                SNAPSHOT_API,
                json={"query": query, "variables": variables},
                timeout=timeout,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 502, 503, 504) and retries < 5:
                ra = r.headers.get("Retry-After")
                delay = float(ra) if (ra and ra.isdigit()) else (backoff**retries)
                time.sleep(delay + random.uniform(0.05, 0.25))
                retries += 1
                continue
            r.raise_for_status()
        except requests.RequestException:
            if retries < 5:
                time.sleep((backoff**retries) + random.uniform(0.05, 0.25))
                retries += 1
                continue
            raise


def _fetch_result_by_id(pid: str) -> Dict[str, Any]:
    data = _gql(PROPOSAL_RESULT_Q, {"id": pid})
    return (data.get("data") or {}).get("proposal") or {}


def _fetch_meta_by_id(pid: str) -> Dict[str, Any]:
    data = _gql(PROPOSAL_BY_ID_Q, {"id": pid})
    return (data.get("data") or {}).get("proposal") or {}


def _fetch_votes_all(pid: str, batch: int = 1000) -> List[dict]:
    out: List[dict] = []
    skip = 0
    while True:
        data = _gql(
            VOTES_BY_PROPOSAL_Q, {"proposal": pid, "first": batch, "skip": skip}
        )
        chunk = (data.get("data") or {}).get("votes") or []
        if not chunk:
            break
        out.extend(chunk)
        if len(chunk) < batch:
            break
        skip += batch
    return out


def _fetch_all_proposals_by_space(space: str, batch: int = 100) -> List[Dict[str, Any]]:
    out: List[dict] = []
    skip = 0
    while True:
        data = _gql(
            PROPOSALS_BY_SPACE_Q, {"space": space, "first": batch, "skip": skip}
        )
        chunk = (data.get("data") or {}).get("proposals") or []
        if not chunk:
            break
        out.extend(chunk)
        if len(chunk) < batch:
            break
        skip += batch
    return out


def _iso_from_unix(ts: Optional[int]) -> Optional[str]:
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Registry/URL helpers
# ----------------------------------------------------------------------------

_DEFILLAMA_SLUG_RE = re.compile(r"/protocol/([a-z0-9-]+)")


def _normalize_snapshot_url(u: str) -> str:
    v = u.replace("/#/s:", "/#/")
    v = re.sub(r"^https://snapshot\\.box", "https://snapshot.org", v)
    return v


def _extract_proposal_id(u: str) -> Optional[str]:
    m = re.search(r"/proposal/(0x[a-fA-F0-9]{64})", u)
    return m.group(1) if m else None


def _resolve_pid_from_url(url: str) -> Optional[str]:
    return _extract_proposal_id(_normalize_snapshot_url(url))


def _space_from_url(snapshot_url: str) -> Optional[str]:
    if not snapshot_url:
        return None
    m = re.search(r"#/s?:?([^/]+)/proposal/", snapshot_url)
    return m.group(1) if m else None


def _extract_token_address_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    def dig(node: Any) -> Optional[str]:
        if isinstance(node, dict):
            for k in ("address", "tokenAddress", "token_address"):
                v = node.get(k)
                if isinstance(v, str) and v.startswith("0x") and len(v) >= 42:
                    return v
            for v in node.values():
                r = dig(v)
                if r:
                    return r
        elif isinstance(node, list):
            for v in node:
                r = dig(v)
                if r:
                    return r
        return None

    space = meta.get("space") or {}
    addr = dig(space.get("strategies"))
    if addr:
        return addr
    return dig(meta.get("strategies"))


# ----------------------------------------------------------------------------
# Adjacent proposals (for context)
# ----------------------------------------------------------------------------

_STOP = set(
    "the and for with from into that this have has are were was will shall of in on to by at as is it be or an a we you they our their".split()
)


def _tokens(text: Optional[str]) -> set:
    if not text:
        return set()
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", text.lower())
    return set(t for t in toks if len(t) >= 3 and t not in _STOP)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def _select_adjacent_proposals(
    all_props: List[Dict[str, Any]],
    current_start_unix: int,
    *,
    max_days: int = 60,
    max_n: int = 10,
    jaccard_min: float = 0.30,
    current_author: Optional[str] = None,
    current_title: Optional[str] = None,
    current_body: Optional[str] = None,
) -> List[Dict[str, Any]]:
    closed_before = [
        p
        for p in all_props
        if (
            p.get("state") == "closed"
            and int(p.get("end") or 0) < int(current_start_unix)
        )
    ]
    closed_before.sort(key=lambda p: int(p.get("end") or 0), reverse=True)
    day_sec = 86400
    within_days = []
    for p in closed_before:
        if (int(current_start_unix) - int(p.get("end") or 0)) <= (max_days * day_sec):
            within_days.append(p)
        else:
            break
    cur_tok = _tokens((current_title or "") + " " + (current_body or ""))

    def _match_topic_or_author(p: Dict[str, Any]) -> bool:
        if (
            current_author
            and p.get("author")
            and str(p["author"]).lower() == str(current_author).lower()
        ):
            return True
        sim = _jaccard(
            cur_tok, _tokens((p.get("title") or "") + " " + (p.get("body") or ""))
        )
        return sim >= jaccard_min

    filtered = [p for p in within_days if _match_topic_or_author(p)]
    base = filtered if filtered else within_days
    take_k = min(len(within_days), max_n) if within_days else min(len(base), max_n)
    return base[:take_k]


# ----------------------------------------------------------------------------
# Impact helpers
# ----------------------------------------------------------------------------


def _pct_change(pre_vals: List[float], post_vals: List[float]) -> Optional[float]:
    pre_vals = [v for v in pre_vals if isinstance(v, (int, float))]
    post_vals = [v for v in post_vals if isinstance(v, (int, float))]
    if not pre_vals or not post_vals:
        return None
    pre_avg = sum(pre_vals) / len(pre_vals)
    post_avg = sum(post_vals) / len(post_vals)
    if pre_avg <= 0:
        return None
    return (post_avg / pre_avg - 1.0) * 100.0


def _resolve_parquet_path(project_root: Path) -> Path:
    envp = os.getenv("CMC_OFFLINE_PARQUET", "").strip()
    if envp:
        p = Path(envp)
        if not p.is_absolute():
            p = project_root / envp
        if p.exists():
            print(f"[price_impact] using parquet from ENV: {p}")
            return p
        else:
            print(f"[price_impact] ENV parquet not found: {p}")
    p = project_root / "cmc_historical_daily_2013_2025.parquet"
    print(f"[price_impact] default parquet: {p} exists={p.exists()}")
    return p


def _normalize_ucid(ucid: Optional[str]) -> Optional[str]:
    if not ucid:
        return None
    s = str(ucid).strip()
    s = re.sub(r"\.0$", "", s)
    if s.isdigit():
        return str(int(s))
    return s or None


def compute_token_price_impact_from_parquet(
    parquet_path: Path,
    ucid: str,
    event_end_utc: str,
    pre_days: int = 3,
    post_days: int = 3,
) -> Optional[float]:
    if not parquet_path.exists():
        print(f"[price_impact] parquet not found: {parquet_path}")
        return None
    try:
        df = pd.read_parquet(
            parquet_path, engine="pyarrow", columns=["date", "ucid", "price_USD"]
        )
    except Exception as e1:
        print(f"[price_impact] pyarrow failed: {e1}")
        try:
            df = pd.read_parquet(
                parquet_path,
                engine="fastparquet",
                columns=["date", "ucid", "price_USD"],
            )
        except Exception as e2:
            print(f"[price_impact] fastparquet failed: {e2}")
            return None

    df = df[["date", "ucid", "price_USD"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["ucid"] = df["ucid"].astype(str).str.strip()
    df["price_USD"] = pd.to_numeric(df["price_USD"], errors="coerce")

    ucid_clean = _normalize_ucid(ucid)
    df_tok = df[df["ucid"] == (ucid_clean or "")]
    if df_tok.empty and ucid_clean and re.fullmatch(r"\d+", ucid_clean):
        df_tok = df[df["ucid"] == str(int(ucid_clean))]

    if df_tok.empty:
        print(f"[price_impact] no rows for ucid={ucid_clean}")
        return None

    try:
        event_date = pd.to_datetime(event_end_utc).date()
    except Exception:
        print(f"[price_impact] bad event_end_utc: {event_end_utc}")
        return None

    pre_start = event_date - timedelta(days=pre_days)
    post_end = event_date + timedelta(days=post_days)
    pre_mask = (df_tok["date"] < event_date) & (df_tok["date"] >= pre_start)
    post_mask = (df_tok["date"] >= event_date) & (df_tok["date"] <= post_end)
    pre_vals = df_tok.loc[pre_mask, "price_USD"].dropna().tolist()
    post_vals = df_tok.loc[post_mask, "price_USD"].dropna().tolist()

    if not pre_vals:
        prev = (
            df_tok[df_tok["date"] < event_date]
            .sort_values("date")
            .tail(1)["price_USD"]
            .dropna()
            .tolist()
        )
        pre_vals = prev or pre_vals
    if not post_vals:
        nxt = (
            df_tok[df_tok["date"] >= event_date]
            .sort_values("date")
            .head(1)["price_USD"]
            .dropna()
            .tolist()
        )
        post_vals = nxt or post_vals

    impact = _pct_change(pre_vals, post_vals)
    return round(impact, 4) if impact is not None else None


def compute_tvl_impact_from_defillama_tool(
    tools,
    *,
    link: Optional[str],
    slug: Optional[str],
    project_hint: Optional[str],
    event_end_utc: str,
    pre_days: int = 3,
    post_days: int = 3,
) -> Optional[float]:
    evt_js = None
    print(f"[defillama] try link={link} slug={slug} project_hint={project_hint}")
    if link:
        _ = _invoke_tool(tools, "refresh_by_link", defillama_link=link)
        evt_js = _invoke_tool(
            tools,
            "event_window_by_link",
            defillama_link=link,
            event_time_utc=event_end_utc,
            pre_days=pre_days,
            post_days=post_days,
        )
    if evt_js is None and slug:
        _ = _invoke_tool(tools, "refresh_protocol", slug=slug)
        evt_js = _invoke_tool(
            tools,
            "event_window",
            slug=slug,
            event_time_utc=event_end_utc,
            pre_days=pre_days,
            post_days=post_days,
        )
    if evt_js is None and project_hint:
        res = _invoke_tool(
            tools, "resolve_protocol", project_hint=project_hint, ttl_hours=0
        )
        cands = (res or {}).get("candidates") or []
        sl = next((c.get("slug") for c in cands if c.get("slug")), None)
        if sl:
            _ = _invoke_tool(tools, "refresh_protocol", slug=sl)
            evt_js = _invoke_tool(
                tools,
                "event_window",
                slug=sl,
                event_time_utc=event_end_utc,
                pre_days=pre_days,
                post_days=post_days,
            )

    if not isinstance(evt_js, dict):
        print("[tvl_impact] event_window returned non-dict or None.")
        return None

    stats = evt_js.get("stats") or {}
    abn = stats.get("abnormal_change")
    if abn is None:
        print("[tvl_impact] no abnormal_change in stats.")
        return None
    try:
        return round(float(abn) * 100.0, 4)
    except Exception:
        return None


def compute_price_impacts_for_adjacent(
    cmc_parquet: Path,
    ucid: Optional[str],
    adjacent: List[Dict[str, Any]],
    pre_days: int = 3,
    post_days: int = 3,
) -> List[Dict[str, Any]]:
    out = []
    if not ucid:
        print("[adj_price] skipped: no UCID")
        return out
    for p in adjacent:
        eid = p.get("end")
        end_iso = _iso_from_unix(eid) if eid else None
        if not end_iso:
            continue
        imp = compute_token_price_impact_from_parquet(
            cmc_parquet, ucid, end_iso, pre_days, post_days
        )
        out.append(
            {"id": p.get("id"), "end_utc": end_iso, "token_price_impact_pct": imp}
        )
    return out


# ----------------------------------------------------------------------------
# Deterministic Timeline analysis via MCP
# ----------------------------------------------------------------------------


def _compute_timeline_via_mcp(
    tools, *, start_ts: int, end_ts: int, choices: List[str], pid: str
) -> Optional[Dict[str, Any]]:
    # 1) Fetch full votes via the snapshot tool if present
    votes_js = None
    snap_names = ["get_votes_all", "snapshot.get_votes_all", "get_votes", "votes_all"]
    snap_params = [{"proposal_id": pid}, {"proposal": pid}, {"id": pid}]
    votes_js = _invoke_tool_try_names_and_params(tools, snap_names, snap_params)
    votes = None
    if isinstance(votes_js, dict):
        votes = votes_js.get("votes") or votes_js.get("data") or votes_js.get("items")
    if not isinstance(votes, list):
        print("[timeline] Fallback to orchestrator GraphQL for votes…")
        votes = _fetch_votes_all(pid)

    try:
        votes.sort(key=lambda v: int(v.get("created", 0)))  # type: ignore
    except Exception:
        pass

    tl_names = ["analyze_timeline", "timeline.analyze_timeline"]
    args = {
        "start": int(start_ts),
        "end": int(end_ts),
        "choices": list(choices or []),
        "votes": list(votes or []),
    }
    tl = _invoke_tool_try_names_and_params(tools, tl_names, [args])
    if not isinstance(tl, dict):
        print("[timeline] analyze_timeline failed or returned non-dict")
        return None
    return tl


# ----------------------------------------------------------------------------
# Forums sentiment via LLM (fetch via ForumsMCP, summarize with LLM)
# ----------------------------------------------------------------------------


def _limit_posts_for_prompt(
    posts: List[Dict[str, Any]], max_chars: int = 15000
) -> List[Dict[str, Any]]:
    out, used = [], 0
    for p in posts:
        raw = p.get("raw") or ""
        used += len(raw)
        out.append(
            {
                "id": p.get("id"),
                "username": p.get("username"),
                "raw": raw,
                "created_at": p.get("created_at"),
            }
        )
        if used >= max_chars:
            break
    return out


def _forum_sentiment_with_llm(
    llm, thread_title: str, posts: List[Dict[str, Any]]
) -> ForumSentiment:
    sys = (
        "You are an expert governance analyst. Read forum comments and produce a JSON with: "
        "overall_polarity (-1..1), stance_share for {'for','against','neutral'} that sum ~1, "
        "toxicity_flags (list of short labels if any), key_topics (<=8), top_positive_quotes (<=3 short quotes), "
        "top_negative_quotes (<=3 short quotes), and summary (<=140 words)."
    )
    user = "Thread title: " + (
        thread_title or ""
    ) + "\n\n" "Posts (id, username, text):\n" + json.dumps(posts, ensure_ascii=False)
    agent = AG(
        atype=ForumSentiment,
        tools=[],
        max_iter=1,
        verbose_agent=False,
        description="Forum sentiment summarizer",
        instructions=sys,
        llm=llm,
    )
    res = asyncio.run(agent << [user])
    try:
        return res.states[0]  # type: ignore
    except Exception:
        return ForumSentiment(summary="Sentiment analysis unavailable.")


def compute_forum_sentiment_if_available(
    tools, discussion_url: Optional[str], llm
) -> Optional[ForumSentiment]:
    if not discussion_url:
        return None
    fetch = _invoke_tool(tools, "fetch_discussion", url=discussion_url, max_pages=5)
    if not isinstance(fetch, dict) or (fetch.get("type") != "discourse"):
        return None
    posts: List[Dict[str, Any]] = fetch.get("posts") or []
    header = fetch.get("thread") or {}
    posts_small = _limit_posts_for_prompt(posts, max_chars=15000)
    sent = _forum_sentiment_with_llm(llm, header.get("title") or "", posts_small)
    if sent:
        try:
            sent.post_count = len(posts_small)
        except Exception:
            pass
    return sent


# ----------------------------------------------------------------------------
# Focus UI (interactive)
# ----------------------------------------------------------------------------

FOCUS_EXPLANATION = (
    "You may specify Focus Areas or Concerns for this proposal (optional).\n"
    "Examples:\n"
    "  1) Token distribution and concentration\n"
    "  2) Treasury impact and budget use\n"
    "  3) Delegate turnout and voter participation\n"
    "  4) Protocol risk (security, liquidity, TVL)\n"
    "  5) Market impact (token price, volatility, trading volume)\n"
    "  6) Governance process quality (discussion sentiment, delegate alignment)\n"
    "  7) Long-term sustainability vs. short-term incentives\n"
)


def _interactive_inputs() -> tuple[str, str]:
    step = "url"
    snapshot_url = ""
    focus = ""
    options = {
        "1": "Token distribution and concentration",
        "2": "Treasury impact and budget use",
        "3": "Delegate turnout and voter participation",
        "4": "Protocol risk (security, liquidity, TVL)",
        "5": "Market impact (token price, volatility, trading volume)",
        "6": "Governance process quality (discussion sentiment, delegate alignment)",
        "7": "Long-term sustainability vs. short-term incentives",
    }
    while True:
        if step == "url":
            val = input("Snapshot Proposal URL> ").strip()
            if not val:
                print("A Snapshot proposal URL is required to continue. ('q' to quit)")
                continue
            if val.lower() == "q":
                raise SystemExit(0)
            snapshot_url = val
            step = "focus_yn"
            continue
        if step == "focus_yn":
            print("\n" + FOCUS_EXPLANATION + "\n")
            yn = (
                input(
                    "Would you like to choose one of the examples above? (y/n, b=back to URL, q=quit) "
                )
                .strip()
                .lower()
            )
            if yn in {"q", "quit"}:
                raise SystemExit(0)
            if yn in {"b", "back"}:
                step = "url"
                continue
            if yn in {"y", "yes"}:
                step = "focus_select"
                continue
            if yn in {"n", "no"}:
                txt = input(
                    "Focus areas or concerns (optional; Enter to skip, 'b' to go back, 'q' to quit)> "
                ).strip()
                if txt.lower() in {"q", "quit"}:
                    raise SystemExit(0)
                if txt.lower() in {"b", "back"}:
                    step = "focus_yn"
                    continue
                focus = txt
                break
            print("Please answer with 'y' or 'n' (or 'b' to go back, 'q' to quit).")
            continue
        if step == "focus_select":
            print(
                "Enter number(s) (comma-separated) or your own custom text. e.g., '1,3' or 'Protocol security review focus'"
            )
            raw = input("Your selection (b=back, q=quit)> ").strip()
            if raw.lower() in {"q", "quit"}:
                raise SystemExit(0)
            if raw.lower() in {"b", "back"}:
                step = "focus_yn"
                continue
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if parts and all(p in options for p in parts):
                focus = "; ".join(options[p] for p in parts)
            else:
                focus = raw
            break
    return snapshot_url, focus


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    load_dotenv()
    project_root = Path(__file__).resolve().parent.parent
    registry_csv = (
        project_root / "src" / "agentics" / "assets_registry" / "dao_registry.csv"
    )

    with ExitStack() as stack:
        all_tools = _load_mcp_tools(project_root, stack)
        tools_for_agent = _blind_toolset(all_tools)

        try:
            llm = get_llm_provider()
        except ValueError as exc:
            raise SystemExit(
                "No LLM provider configured. Populate .env or env vars."
            ) from exc

        # --- Inputs ---
        snapshot_url, focus = _interactive_inputs()
        pid = _resolve_pid_from_url(snapshot_url)
        if not pid:
            raise SystemExit("Could not resolve proposal id from URL.")

        # --- Snapshot authoritative timing & metadata ---
        result_js = _fetch_result_by_id(pid)
        if not result_js:
            raise SystemExit("Snapshot RESULT empty. Check network or proposal id.")
        start_ts = result_js.get("start")
        end_ts = result_js.get("end")
        if start_ts is None or end_ts is None:
            raise SystemExit("Snapshot RESULT missing `start`/`end` UNIX timestamps.")
        start_iso = _iso_from_unix(start_ts)
        end_iso = _iso_from_unix(end_ts)
        if not start_iso or not end_iso:
            raise SystemExit(
                "Invalid `start`/`end` UNIX timestamps in Snapshot RESULT."
            )

        meta_js = _fetch_meta_by_id(pid) or {}
        choices: List[str] = (
            (meta_js.get("choices") or []) if isinstance(meta_js, dict) else []
        )
        discussion_url: Optional[str] = (
            (meta_js.get("discussion") or None) if isinstance(meta_js, dict) else None
        )

        # --- Deterministic timeline metrics via MCP (guaranteed) ---
        timeline_out = _compute_timeline_via_mcp(
            all_tools, start_ts=start_ts, end_ts=end_ts, choices=choices, pid=pid
        )
        if timeline_out:
            print("\n=== Timeline (via MCP) ===")
            print(
                json.dumps(
                    {
                        k: timeline_out.get(k)
                        for k in [
                            "unique_voters",
                            "total_votes",
                            "first_vote_at",
                            "last_vote_at",
                            "series_step_hours",
                        ]
                    },
                    indent=2,
                )
            )

        # --- Forum sentiment (LLM‑driven, optional) ---
        forum_sentiment = compute_forum_sentiment_if_available(
            all_tools, discussion_url, llm
        )
        if forum_sentiment:
            print("\n=== Forum Sentiment (LLM) ===")
            print(_to_json_str(forum_sentiment, indent=2))

            def _top3(lst):
                return [q for q in (lst or [])][:3]

            print(f"Posts analyzed: {forum_sentiment.post_count or 0}")
            print("Top + quotes:")
            print(
                json.dumps(
                    _top3(forum_sentiment.top_positive_quotes),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            print("Top - quotes:")
            print(
                json.dumps(
                    _top3(forum_sentiment.top_negative_quotes),
                    ensure_ascii=False,
                    indent=2,
                )
            )

        # --- Adjacent proposal baseline (for context) ---
        space_hint = _space_from_url(snapshot_url) or (meta_js.get("space") or {}).get(
            "id"
        )
        adjacent_list: List[Dict[str, Any]] = []
        if space_hint:
            try:
                all_props = _fetch_all_proposals_by_space(space_hint)
            except Exception:
                all_props = []
            adjacent_list = _select_adjacent_proposals(
                all_props,
                current_start_unix=int(start_ts),
                max_days=60,
                max_n=10,
                jaccard_min=0.30,
                current_author=meta_js.get("author"),
                current_title=meta_js.get("title"),
                current_body=meta_js.get("body"),
            )

        def _fmt_adj(p: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "id": p.get("id"),
                "title": p.get("title"),
                "author": p.get("author"),
                "start": int(p.get("start") or 0),
                "end": int(p.get("end") or 0),
                "end_utc": _iso_from_unix(p.get("end")) if p.get("end") else None,
                "state": p.get("state"),
                "choices_len": len(p.get("choices") or []),
            }

        ADJACENT_BASELINE = list(map(_fmt_adj, adjacent_list))
        print(
            "\n=== Adjacent Past Proposals (≤10 within 60d & topic/author matched) ==="
        )
        print(json.dumps(ADJACENT_BASELINE, ensure_ascii=False, indent=2))

        # --- Price/TVL impacts ---
        cmc_parquet = _resolve_parquet_path(project_root)

        # Registry CSV (space → ucid/defillama hints)
        cmc_ucid = None
        defillama_link = None
        defillama_slug = None
        if registry_csv and space_hint and registry_csv.exists():
            try:
                with registry_csv.open("r", encoding="utf-8-sig") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        if (
                            str(row.get("space") or "").strip().lower()
                            == str(space_hint).strip().lower()
                        ):
                            cmc_ucid = (
                                str(
                                    row.get("cmc_ucid") or row.get("ucid") or ""
                                ).strip()
                                or None
                            )
                            defillama_link = (
                                row.get("defillama_link") or ""
                            ).strip() or None
                            name_defi = (
                                row.get("name_defillama")
                                or row.get("defillama_name")
                                or ""
                            ).strip()
                            if not defillama_link and name_defi:
                                slug = re.sub(r"\s+", "-", name_defi)
                                slug = re.sub(r"[^a-zA-Z0-9\-]", "", slug).lower()
                                defillama_slug = slug or None
                            if defillama_link and not defillama_slug:
                                m = _DEFILLAMA_SLUG_RE.search(defillama_link.lower())
                                defillama_slug = m.group(1) if m else None
                            break
            except Exception as e:
                print(f"[registry] read failed: {e}")

        cmc_ucid = _normalize_ucid(cmc_ucid)
        print(
            f"[registry] space={space_hint} cmc_ucid={cmc_ucid} defillama_link={defillama_link} slug={defillama_slug}"
        )

        token_price_impact = None
        if cmc_ucid and cmc_parquet.exists():
            token_price_impact = compute_token_price_impact_from_parquet(
                cmc_parquet, cmc_ucid, end_iso, 3, 3
            )

        tvl_impact = compute_tvl_impact_from_defillama_tool(
            all_tools,
            link=defillama_link if defillama_link else None,
            slug=defillama_slug if (defillama_slug and not defillama_link) else None,
            project_hint=space_hint,
            event_end_utc=end_iso,
            pre_days=3,
            post_days=3,
        )
        print("\n=== Impact Preview (current) ===")
        print(
            json.dumps(
                {
                    "token_price_impact_pct": token_price_impact,
                    "tvl_impact_pct": tvl_impact,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        # Past proposals price impact
        adj_price_impacts = compute_price_impacts_for_adjacent(
            cmc_parquet, cmc_ucid, adjacent_list, 3, 3
        )
        if adj_price_impacts:
            print("\n=== Past Proposals — Token Price Impact (±3d) ===")
            print(json.dumps(adj_price_impacts, indent=2, ensure_ascii=False))

        # --- Agent pass: plan ---
        votes_count = (timeline_out or {}).get("total_votes", 0)
        plan_prompt = [
            f"SNAPSHOT_URL: {snapshot_url}",
            f"PROPOSAL_ID: {pid}",
            f"BOOTSTRAP_CHOICES: {choices}",
            f"DISCUSSION_URL: {discussion_url}",
            f"EVENT_START_UTC: {start_iso}",
            f"EVENT_END_UTC: {end_iso}",
            f"TIMELINE_ALREADY_COMPUTED: {bool(timeline_out)} (use for context)",
            f"VOTES_TOTAL_OBSERVED: {votes_count}",
            "OBJECTIVE: Determine one choice to recommend (ex-post blind).",
            "MANDATORY_CALLS (for reproducibility):\n- snapshot.get_votes_all(proposal_id) → timeline.analyze_timeline(start,end,choices,votes).",
        ]
        if focus:
            plan_prompt.append(f"EXTRA EMPHASIS: {focus}")

        plan_agent = AG(
            atype=ToolPlan,
            tools=tools_for_agent,
            max_iter=4,
            verbose_agent=False,
            description="Plan which MCP tools to call for a governance evaluation.",
            instructions="Return a ToolPlan JSON only; do not include prose outside JSON.",
            llm=get_llm_provider(),
        )
        plan_result = asyncio.run(plan_agent << ["\n".join(plan_prompt)])
        try:
            tool_plan: ToolPlan = plan_result.states[0]  # type: ignore
        except Exception:
            tool_plan = ToolPlan(objective="(plan unavailable)", calls=[], notes=None)
        print("\n=== ToolPlan (agent drafted) ===")
        print(_to_json_str(tool_plan, indent=2))

        # --- Agent pass: decision ---
        decision_prompt = [
            f"Snapshot proposal under review: {snapshot_url}",
            f"Authoritative context: choices={choices}, discussion={discussion_url}, event_start_utc={start_iso}, event_end_utc={end_iso}",
            "Use the provided previews, but you may re-call tools for verification.",
            "Objective: Choose exactly one option from the proposal's `choices`.\nFill every field of ProposalDecision. Do NOT use ex-post tally.",
            "ONCHAIN GOAL: total holders & top-100 concentration (you may call onchain/holders MCPs if configured).",
            f"PLANNED CALLS: {_to_json_str(tool_plan)}",
        ]
        if focus:
            decision_prompt.append(f"Extra emphasis: {focus}")

        decision_agent = AG(
            atype=ProposalDecision,
            tools=tools_for_agent,
            max_iter=14,
            verbose_agent=False,
            description="Governance vote recommendation for a Snapshot proposal (ex-post blind).",
            instructions=(
                "Return a ProposalDecision object. Use the provided BOOTSTRAP_CONTEXT. "
                "Choose exactly one option from `choices` and set both label and index. "
                "Include the full `choices` in available_choices. "
                "Set event_start_utc and event_end_utc (copy end into event_time_utc). "
                "Set address_of_governance_token if known."
            ),
            llm=get_llm_provider(),
        )
        decision_result = asyncio.run(decision_agent << ["\n".join(decision_prompt)])
        print("\n=== ProposalDecision ===")
        try:
            print(decision_result.pretty_print())
        except Exception:
            states_tmp = getattr(decision_result, "states", [])
            if states_tmp:
                print(_to_json_str(states_tmp[0], indent=2))  # type: ignore

        states = getattr(decision_result, "states", [])
        if not states:
            raise SystemExit("Agent produced no decision state; cannot write log.")
        decision: ProposalDecision = states[0]  # type: ignore

        # --- Coercions & enrichment ---
        def _coerce_choice(
            label: str, idx: Optional[int], options: List[str]
        ) -> Tuple[str, Optional[int]]:
            if not options:
                return (label, idx)
            for i, c in enumerate(options):
                if (label or "") == (c or ""):
                    return (c, i)
            norm = (label or "").strip().lower()
            for i, c in enumerate(options):
                if norm == (c or "").strip().lower():
                    return (c, i)
            if idx is not None and 0 <= idx < len(options):
                return (options[idx], idx)
            return (label, idx)

        decision.selected_choice_label, decision.selected_choice_index = _coerce_choice(
            decision.selected_choice_label, decision.selected_choice_index, choices
        )
        decision.available_choices = choices or decision.available_choices
        decision.event_start_utc = start_iso
        decision.event_end_utc = end_iso
        decision.event_time_utc = end_iso

        # Persist our computed artifacts
        if timeline_out and not decision.timeline_metrics:
            decision.timeline_metrics = timeline_out
        if forum_sentiment and not decision.forum_sentiment:
            decision.forum_sentiment = forum_sentiment

        # Inject computed impacts
        if token_price_impact is not None:
            decision.token_price_impact_pct = token_price_impact
        if tvl_impact is not None:
            decision.tvl_impact_pct = tvl_impact

        # Ex‑post winner/margins from RESULT (orchestrator‑side only)
        scores = result_js.get("scores") or []
        scores_total = result_js.get("scores_total")
        actual = ActualVoteResult()
        actual.scores = [float(x) for x in scores] if scores else None
        actual.scores_total = float(scores_total) if scores_total is not None else None
        if scores and choices:
            winner_idx = max(range(len(scores)), key=lambda i: scores[i])
            actual.winner_index = winner_idx
            actual.winner_label = (
                choices[winner_idx] if 0 <= winner_idx < len(choices) else None
            )
            sorted_scores = sorted(scores, reverse=True)
            if len(sorted_scores) >= 2 and actual.scores_total:
                actual.margin_abs = float(sorted_scores[0] - sorted_scores[1])
                actual.margin_pct = round(
                    float(actual.margin_abs / actual.scores_total), 6
                )
        decision.actual_vote_result = actual

        # --- Save structured record ---
        payload = {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_url": snapshot_url,
            "focus": focus or None,
            "tool_plan": _to_dict(tool_plan),
            "decision": _to_dict(decision),
        }
        decision_dir = project_root / "Decision_runs"
        decision_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot_slug = snapshot_url.strip().split("/")[-1] or "proposal"
        outfile = decision_dir / f"decision_{snapshot_slug}_{timestamp}.json"
        with outfile.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nSaved structured decision to {outfile}")


if __name__ == "__main__":
    main()
