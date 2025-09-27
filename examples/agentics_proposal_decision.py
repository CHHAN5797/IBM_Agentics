# ===============================
# File: agentics_proposal_decision.py  (Orchestrator, UPDATED+EXTENDED)
# ===============================
from __future__ import annotations

"""
Interactive Agentics example — governance vote recommendation (ex-post blind planning)
with Focus Areas, Snapshot RESULT-based event time, DeFiLlama TVL via CSV link/slug, and CMC offline price impact.

This version includes:
- (1) Robust MCP result unwrapping (incl. list[str] → JSON restore) and stable `get_votes_all`.
- (2) One-time semantic literature bootstrap (DAO governance, token distribution, voting games, info aggregation).
- (3) Adjacent proposals include an explicit Jaccard similarity score.
- (4) ProposalDecision fields `ai_final_conclusion`, `ai_final_reason`, and stronger prompt to fill them.
- (5) Agent-driven forum analytics - agent calls forum/sentiment tools when needed instead of pre-processing.
"""
import csv
import html as _html_mod
import json
import os
import random
import re
import re as _re_mod
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from crewai_tools import MCPServerAdapter
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agentics.core.llm_connections import get_llm_provider
from agentics.orchestra.dual_llm_runner import run_both_and_save
from mcp import StdioServerParameters

from decision_agent_runner import (
    ActualVoteResult,
    DecisionAgentContext,
    run_decision_agent,
)

# --- CrewAI FilteredStream flush guard ---
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


def _html_to_text(html: str) -> str:
    if not html:
        return ""
    html = _re_mod.sub(r"(?i)<\s*br\s*/?\s*>", "\n", html)
    html = _re_mod.sub(r"(?i)</\s*p\s*>", "\n\n", html)
    text = _re_mod.sub(r"<[^>]+>", "", html)
    return _html_mod.unescape(text).strip()


def decide_and_archive(prompt_blocks):
    saved = run_both_and_save(
        messages=prompt_blocks,
        decision_root="Decision_runs",
        openai_model="gpt-4o-mini",
        grok_model="grok-2",
        temperature=0.2,
    )
    return saved


# ------------------------------------------------------------------------------
# Pydantic helpers
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# MCP wiring
# ------------------------------------------------------------------------------


@dataclass
class ServerSpec:
    name: str
    script_path: Optional[Path] = None
    module_name: Optional[str] = None
    extra_env: Optional[Dict[str, str]] = None

def _server_params(spec: ServerSpec, src_dir: Path) -> StdioServerParameters:
    env = {"PYTHONPATH": str(src_dir), **os.environ}
    if spec.extra_env:
        env.update(spec.extra_env)

    if spec.module_name:
        args = ["-m", spec.module_name]
    elif spec.script_path:
        args = [str(spec.script_path)]
    else:
        raise ValueError(f"ServerSpec '{spec.name}' missing entrypoint")

    return StdioServerParameters(command="python3", args=args, env=env)


def _load_mcp_tools(project_root: Path, stack: ExitStack) -> List:
    src_dir = project_root / "src"
    cmc_parquet = project_root / "cmc_historical_daily_2013_2025.parquet"
    cmc_env = (
        {"CMC_OFFLINE_PARQUET": str(cmc_parquet)} if cmc_parquet.exists() else None
    )

    server_specs = [
        ServerSpec(
            name="snapshot",
            script_path=project_root / "src/agentics/mcp/snapshot_api.py",
        ),
        ServerSpec(
            name="timeline",
            script_path=project_root / "src/agentics/mcp/timeline_mcp.py",
        ),
        ServerSpec(
            name="forums",
            script_path=project_root / "src/agentics/mcp/forums_mcp.py",
        ),
        ServerSpec(
            name="defillama",
            module_name="agentics.mcp.defillama_mcp",
        ),
        ServerSpec(
            name="holders",
            script_path=project_root / "src/agentics/mcp/holders_activity_mcp.py",
        ),
        ServerSpec(
            name="onchain",
            script_path=project_root / "src/agentics/mcp/onchain_activity_mcp_bq_cmc.py",
        ),
        ServerSpec(
            name="cmc",
            script_path=project_root / "src/agentics/mcp/cmc_offline_mcp.py",
            extra_env=cmc_env,
        ),
        ServerSpec(
            name="semantics",
            script_path=project_root / "src/agentics/mcp/semantics_mcp.py",
        ),
        ServerSpec(
            name="registry",
            script_path=project_root / "src/agentics/mcp/registry_mcp.py",
        ),
        ServerSpec(
            name="govnews",
            script_path=project_root / "src/agentics/mcp/govnews_mcp.py",
        ),
        ServerSpec(
            name="sentiment",
            script_path=project_root / "src/agentics/mcp/sentiment_mcp.py",
        ),
    ]

    combined = None
    for spec in server_specs:
        if spec.script_path and not spec.script_path.exists():
            print(f"[skip] {spec.name}: missing server at {spec.script_path}")
            continue
        if not spec.script_path and not spec.module_name:
            print(f"[skip] {spec.name}: missing module/script configuration")
            continue
        adapter = stack.enter_context(
            MCPServerAdapter(_server_params(spec, src_dir))
        )
        entry = spec.module_name or str(spec.script_path)
        print(f"Available {spec.name} ({entry}) tools: {[tool.name for tool in adapter]}")
        combined = adapter if combined is None else combined + adapter
    if combined is None:
        raise RuntimeError(
            "No MCP servers could be started. Check dependencies and configuration."
        )
    return combined


def _blind_toolset(tools) -> List:
    banned = {
        "get_proposal_result_by_id"
    }  # keep blind to ex-post tally for decision pass
    return [t for t in tools if getattr(t, "name", None) not in banned]


# ------------------------------------------------------------------------------
# Robust normalization for MCP / CrewAI return shapes
# ------------------------------------------------------------------------------
def _normalize_tool_result(res):
    """Best-effort: unwrap CrewAI/MCP tool results into plain Python (dict/list).
    Handles dict/list; pydantic-like objects; .content/.data/.result/.output holders;
    raw JSON strings; and list[str] that are JSON lines or split JSON.
    """
    import ast
    import json

    def _json_or_literal(s: str):
        s = (s or "").strip()
        if not s:
            return s
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            return ast.literal_eval(s)
        except Exception:
            return s

    if res is None:
        return None

    if isinstance(res, (dict, list)):
        # list[str] → attempt to parse
        if isinstance(res, list) and all(isinstance(x, str) for x in res):
            joined = "\n".join(res).strip()
            parsed = _json_or_literal(joined)
            if isinstance(parsed, (dict, list)):
                return parsed
            items = []
            for line in res:
                p = _json_or_literal(line)
                if isinstance(p, list):
                    items.extend(p)
                elif isinstance(p, dict):
                    items.append(p)
            return items if items else res
        return res

    for attr in ("model_dump", "dict"):
        fn = getattr(res, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass

    for attr in ("content", "data", "result", "output"):
        if hasattr(res, attr):
            val = getattr(res, attr)
            if isinstance(val, str):
                return _json_or_literal(val)
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                joined = "\n".join(val).strip()
                parsed = _json_or_literal(joined)
                return parsed if isinstance(parsed, (dict, list)) else val
            return val

    if isinstance(res, str):
        return _json_or_literal(res)

    return res


# ------------------------------------------------------------------------------
# Tool invocation helpers
# ------------------------------------------------------------------------------
def _invoke_tool(tools, name: str, **kwargs):
    """Tolerant tool invocation across .call/.run/(**kwargs)/(payload) + result unwrapping."""
    tool = next((t for t in tools if getattr(t, "name", None) == name), None)
    if tool is None:
        print(
            f"[invoke] Tool not found: {name}. Available: {[getattr(t,'name',None) for t in tools]}"
        )
        return None
    payload = dict(kwargs)

    def _try(callable_name, as_kwargs=True):
        fn = getattr(tool, callable_name, None)
        if not fn:
            return None
        try:
            if as_kwargs:
                return fn(**payload)
            else:
                return fn(payload)
        except Exception as e:
            print(
                f"[invoke] {name}.{callable_name}({'**' if as_kwargs else 'dict'}) failed: {e}"
            )
            return None

    res = (
        _try("call", True)
        or _try("run", True)
        or _try("call", False)
        or _try("run", False)
    )
    if res is None and callable(tool):
        try:
            res = tool(**kwargs)
        except Exception as e:
            print(f"[invoke] {name}(**kwargs) failed: {e}")
            res = None

    out = _normalize_tool_result(res)
    typ = type(out).__name__
    preview = (
        (str(out)[:120] + "...")
        if not isinstance(out, (dict, list))
        else f"{typ} (len={len(out) if isinstance(out,list) else 'dict'})"
    )
    print(f"[invoke] {name} -> {typ}: {preview}")
    return out


def _invoke_tool_try_names_and_params(
    tools, names: List[str], param_variants: List[Dict[str, Any]]
) -> Optional[dict | list]:
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


# ------------------------------------------------------------------------------
# Snapshot GraphQL (kept for metadata/result; votes now via MCP)
# ------------------------------------------------------------------------------
SNAPSHOT_API = os.getenv("SNAPSHOT_API", "https://hub.snapshot.org/graphql")
_HTTP = requests.Session()
_HTTP.headers.update({"User-Agent": "agentics-orchestrator/1.3"})

PROPOSAL_RESULT_Q = """
query($id: String!) {
  proposal(id: $id) { id choices scores scores_total state start end }
}
"""

PROPOSAL_BY_ID_Q = """
query ($id: String!) {
  proposal(id: $id) {
    id title body author choices discussion start end state
    space { id name strategies { name network params } }
  }
}
"""

PROPOSALS_BY_SPACE_Q = """
query($space: String!, $first: Int!, $skip: Int!) {
  proposals(first:$first, skip:$skip, where:{ space_in: [$space] },
            orderBy:"created", orderDirection:desc){
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


# ------------------------------------------------------------------------------
# Helpers (url, registry, token addr)
# ------------------------------------------------------------------------------
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


def _pick_project_hint(space: Optional[str], title: Optional[str]) -> Optional[str]:
    if space:
        base = space.split(".")[0].lower()
        base = re.sub(r"(?:-)?(dao|governance|protocol)$", "", base)
        return base
    if title:
        return re.split(r"[\s:]+", title)[0]
    return None


def _extract_token_address_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    def dig(node: Any) -> Optional[str]:
        if isinstance(node, dict):
            for k in ("address", "tokenAddress", "token_address"):
                v = node.get(k)
                if isinstance(v, str) and v.startswith("0x") and len(v) >= 42:
                    return v
            for v in node.values():
                r = dig(v)
                # type: ignore
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


def _norm_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = s.replace("\u200b", "")
    return re.sub(r"\s+", "", s).strip().lower()


def _clean_ucid(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if re.fullmatch(r"\d+\.\d+", s):
        s = s.split(".", 1)[0]
    return s


def _slug_from_defillama_link(link: Optional[str]) -> Optional[str]:
    if not link or not isinstance(link, str):
        return None
    m = re.search(r"/protocol/([a-z0-9-]+)", link.strip().lower())
    return m.group(1) if m else None


def _registry_csv_lookup_space_only(
    csv_path: Path, *, space: Optional[str]
) -> Optional[Dict[str, Any]]:
    if not csv_path or not csv_path.exists():
        print(f"[registry] CSV not found at {csv_path}")
        return None
    try:
        with csv_path.open("r", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            rows_raw = list(reader)
    except Exception:
        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows_raw = list(reader)

    def _split_contracts(s: str) -> List[str]:
        return [x.strip() for x in re.split(r"[|,;]+", s or "") if x.strip()]

    target = _norm_text(space)
    if not target:
        print("[registry] no 'space' provided; cannot match.")
        return None

    spaces_seen = []
    for r in rows_raw:
        low = {
            (k or "").strip().lower(): (v if v is not None else "")
            for k, v in r.items()
        }
        row_space = (low.get("space") or "").strip()
        row_ucid = (low.get("cmc_ucid") or low.get("ucid") or "").strip()
        row_ticker = (low.get("cmc_ticker") or low.get("ticker") or "").strip()
        row_defi = (
            low.get("name_defillama")
            or low.get("defillama_name")
            or low.get("name")
            or ""
        ).strip()
        row_link = (low.get("defillama_link") or "").strip()
        row_contracts_raw = (low.get("contracts") or low.get("contract") or "").strip()

        if row_space:
            spaces_seen.append(row_space)

        if _norm_text(row_space) == target:
            return {
                "space": row_space or None,
                "cmc_ucid": row_ucid or None,
                "cmc_ticker": row_ticker or None,
                "defillama_name": row_defi or None,
                "defillama_link": row_link or None,
                "contracts": _split_contracts(row_contracts_raw) or None,
            }

    print(f"[registry] space not found (space-only): {space!r}")
    print(
        f"[registry] sample spaces in CSV: {sorted(set(spaces_seen))[:10]}{' ...' if len(set(spaces_seen))>10 else ''}"
    )
    return None


# ------------------------------------------------------------------------------
# Adjacent proposals (selection) + similarity
# ------------------------------------------------------------------------------
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
    """Pick past closed proposals near in time and topic/author for contextual comparison."""
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


# ------------------------------------------------------------------------------
# Impact helpers (price & TVL)
# ------------------------------------------------------------------------------
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


def compute_token_price_impact_from_parquet(
    parquet_path: Path,
    ucid: str,
    event_end_utc: str,
    pre_days: int = 7,
    post_days: int = 7,
) -> Optional[float]:
    """Compute average pre/post % change using offline CMC parquet."""
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

    ucid_clean = re.sub(r"\.0$", "", str(ucid).strip())
    df_tok = df[df["ucid"] == ucid_clean]
    if df_tok.empty and re.fullmatch(r"\d+", ucid_clean):
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
    pre_days: int = 7,
    post_days: int = 7,
) -> Optional[float]:
    """
    Strategy:
      1) If link provided → refresh_by_link + event_window_by_link
      2) Else if slug provided → refresh_protocol + event_window
      3) Else if project_hint provided → resolve_protocol(project_hint) → step 2
    Returns abnormal_change * 100 (%), rounded to 4 decimals.
    """
    evt_js = None
    # 1) Link path
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
    # 2) Slug path
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
    # 3) Resolve path
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


# ------------------------------------------------------------------------------
# Focus UI
# ------------------------------------------------------------------------------
FOCUS_EXPLANATION = """
You may specify Focus Areas or Concerns for this proposal (optional).
Examples:
  1) Token distribution and concentration
  2) Treasury impact and budget use
  3) Delegate turnout and voter participation
  4) Protocol risk (security, liquidity, TVL)
  5) Market impact (token price, volatility, trading volume)
  6) Governance process quality (discussion sentiment, delegate alignment)
  7) Long-term sustainability vs. short-term incentives
""".strip()


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


# ------------------------------------------------------------------------------
# NEW: Votes via Snapshot MCP (robust), Timeline call, Forum fetch/summary
# ------------------------------------------------------------------------------
def _get_votes_via_snapshot_mcp(tools, proposal_id: str) -> List[dict]:
    """Fetch full votes array using Snapshot MCP (not GraphQL here)."""
    import ast
    import json

    def _coerce_votes(obj) -> List[dict]:
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            return obj
        if isinstance(obj, dict):
            for k in ("votes", "result", "data"):
                v = obj.get(k)
                if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                    return v
        if isinstance(obj, str):
            try:
                p = json.loads(obj)
            except Exception:
                try:
                    p = ast.literal_eval(obj)
                except Exception:
                    p = None
            return _coerce_votes(p) if p is not None else []
        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            joined = "\n".join(obj).strip()
            try:
                p = json.loads(joined)
                return _coerce_votes(p)
            except Exception:
                pass
            items = []
            for line in obj:
                s = line.strip()
                if not s:
                    continue
                p = None
                try:
                    p = json.loads(s)
                except Exception:
                    try:
                        p = ast.literal_eval(s)
                    except Exception:
                        p = None
                if isinstance(p, dict):
                    items.append(p)
                elif isinstance(p, list):
                    items.extend([x for x in p if isinstance(x, dict)])
            return items
        return []

    names = ["get_votes_all", "snapshot.get_votes_all", "SnapshotAPI.get_votes_all"]
    variants = [
        {"proposal_id": proposal_id},
        {"id": proposal_id},
        {"proposal": proposal_id},
        {"proposal_id": proposal_id, "batch": 1000},
    ]
    res = _invoke_tool_try_names_and_params(tools, names, variants)
    res = _normalize_tool_result(res)

    votes_raw = _coerce_votes(res)
    if not votes_raw:
        print("[votes] could not retrieve votes via MCP; defaulting to empty list.")
        return []

    out = []
    for v in votes_raw:
        if not isinstance(v, dict):
            continue
        out.append(
            {
                "id": v.get("id"),
                "voter": (v.get("voter") or "").lower().strip(),
                "created": int(v.get("created") or 0),
                "choice": v.get("choice"),
                "vp": (
                    float(v.get("vp") or 0.0)
                    if str(v.get("vp") or "").strip() != ""
                    else 0.0
                ),
                "reason": v.get("reason"),
            }
        )
    return out


def _analyze_timeline(
    tools, start_unix: int, end_unix: int, choices: List[str], votes: List[dict]
) -> Dict[str, Any]:
    return (
        _invoke_tool(
            tools,
            "analyze_timeline",
            start=int(start_unix),
            end=int(end_unix),
            choices=choices,
            votes=votes,
        )
        or {}
    )


def _fetch_forum_comments(tools, discussion_url: Optional[str]) -> Dict[str, Any]:
    if not discussion_url:
        return {"url": None, "comments": []}

    pack = _invoke_tool(tools, "fetch_discussion", url=discussion_url) or {}

    raw = pack.get("comments") or pack.get("data") or pack.get("posts") or []
    norm = []
    for c in raw if isinstance(raw, list) else []:
        if isinstance(c, dict):
            author = c.get("author") or c.get("user") or c.get("username")
            created = c.get("created") or c.get("created_at") or c.get("time")
            body = c.get("body") or c.get("content") or c.get("raw") or c.get("text")
            if not body:
                cooked = c.get("cooked")
                if cooked:
                    body = _html_to_text(cooked)
            norm.append(
                {
                    "author": author,
                    "created": created,
                    "body": body or "",
                    "url": c.get("url") or c.get("link"),
                }
            )
        else:
            norm.append({"author": None, "created": None, "body": str(c), "url": None})

    return {"url": discussion_url, "comments": norm}


# --- Forum summary/sentiment (lightweight buckets) ---
def _sentiment_bucket(text: str) -> str:
    t = (text or "").lower()
    pos_kw = (
        "great",
        "good",
        "support",
        "agree",
        "benefit",
        "positive",
        "approve",
        "yes",
        "advantage",
        "+1",
        "in favor",
    )
    neg_kw = (
        "bad",
        "concern",
        "against",
        "disagree",
        "risk",
        "negative",
        "reject",
        "no",
        "harm",
        "-1",
        "oppose",
    )
    pos = any(k in t for k in pos_kw)
    neg = any(k in t for k in neg_kw)
    if pos and not neg:
        return "Positive"
    if neg and not pos:
        return "Negative"
    return "Neutral"


def _summarize_forum_comments(forum_pack: dict, tools=None) -> dict:
    comments = forum_pack.get("comments") or []
    total = len(comments)

    # 최근 5개 (created desc)
    def _key(c):
        try:
            return int(c.get("created") or 0)
        except Exception:
            return 0

    recent = sorted(comments, key=_key, reverse=True)[:5]
    previews = [
        {
            "author": c.get("author"),
            "created": c.get("created"),
            "preview": (c.get("body") or "")[:100],
        }
        for c in recent
    ]

    # 기본값
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

    if tools and comments:
        # ✅ 네임스페이스가 붙은 MCP 이름들도 모두 시도
        res = (
            _invoke_tool_try_names_and_params(
                tools,
                names=[
                    "classify_forum_comments",
                    "sentiment.classify_forum_comments",
                    "SentimentMCP.classify_forum_comments",
                ],
                param_variants=[{"comments": comments}],
            )
            or {}
        )

        # 1) MCP가 정상 응답한 경우
        mcp_counts = (isinstance(res, dict) and res.get("counts")) or None
        if isinstance(mcp_counts, dict) and set(mcp_counts.keys()) >= {
            "Positive",
            "Negative",
            "Neutral",
        }:
            counts = {
                k: int(mcp_counts.get(k, 0))
                for k in ("Positive", "Negative", "Neutral")
            }
        else:

            local = {"Positive": 0, "Negative": 0, "Neutral": 0}
            for c in comments:
                lab = _sentiment_bucket(c.get("body") or "")
                local[lab] = local.get(lab, 0) + 1
            counts = local

    return {
        "total_comments": total,
        "recent_previews": previews,
        "sentiment_counts": counts,
    }


# --- One-time semantic reference bootstrap (cached) ---
def _bootstrap_semantic_references(tools, project_root: Path) -> List[dict]:
    cache = project_root / "Decision_runs" / "semantic_refs.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass

    queries = [
        "DAO governance empirical analysis",
        "on-chain governance token distribution concentration voting power",
        "information aggregation in token-based voting",
        "voting game theory in blockchain governance",
        "delegated voting and participation in DAOs",
    ]

    tool_names = [
        "find_papers",
        "semantics_search",
        "search_papers",
        "literature_search",
        "semantic_search",
    ]

    refs: List[dict] = []

    res = _invoke_tool_try_names_and_params(
        tools,
        tool_names,
        [
            {"queries": queries, "per_query": 5, "enrich": True},
            {"queries": queries, "per_query": 5},
            {"queries": queries},
        ],
    )
    if isinstance(res, list):
        for r in res:
            if isinstance(r, dict):
                refs.append(
                    {
                        "title": r.get("title") or r.get("name"),
                        "url": r.get("url") or r.get("link"),
                        "year": r.get("year") or r.get("date"),
                        "doi": r.get("doi"),
                        "source": r.get("source") or "semantics",
                    }
                )

    if not refs:
        for q in queries:
            res1 = _invoke_tool_try_names_and_params(
                tools,
                tool_names,
                [
                    {"queries": [q], "per_query": 5, "enrich": True},
                    {"queries": [q], "per_query": 5},
                    {"queries": [q]},
                ],
            )
            if isinstance(res1, list):
                for r in res1:
                    if isinstance(r, dict):
                        refs.append(
                            {
                                "title": r.get("title") or r.get("name") or q,
                                "url": r.get("url") or r.get("link"),
                                "year": r.get("year") or r.get("date"),
                                "doi": r.get("doi"),
                                "source": r.get("source") or "semantics",
                            }
                        )

    seen = set()
    uniq = []
    for r in refs:
        k = (r.get("title"), r.get("url"))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    cache.write_text(json.dumps(uniq, ensure_ascii=False, indent=2), encoding="utf-8")
    return uniq


# ------------------------------------------------------------------------------
# Adjacent analytics (compute metrics/impacts) + similarity reporting
# ------------------------------------------------------------------------------
def _adjacent_analytics(
    tools,
    proposals: List[Dict[str, Any]],
    *,
    cmc_parquet: Path,
    ucid: Optional[str],
    link: Optional[str],
    slug: Optional[str],
    project_hint: Optional[str],
    current_title: Optional[str],
    current_body: Optional[str],
    max_items: int = 3,
) -> List[Dict[str, Any]]:
    """For each adjacent proposal, compute timeline metrics and impacts using the same token/TVL handles."""
    out: List[Dict[str, Any]] = []
    cur_tok = _tokens((current_title or "") + " " + (current_body or ""))
    for p in proposals[:max_items]:
        try:
            pid = p.get("id")
            start = int(p.get("start") or 0)
            end = int(p.get("end") or 0)
            choices = p.get("choices") or []
            votes = _get_votes_via_snapshot_mcp(tools, pid) if pid else []
            timeline = (
                _analyze_timeline(tools, start, end, choices, votes) if votes else {}
            )
            end_iso = _iso_from_unix(end)
            price_imp = (
                compute_token_price_impact_from_parquet(cmc_parquet, ucid, end_iso)
                if (ucid and end_iso)
                else None
            )
            tvl_imp = compute_tvl_impact_from_defillama_tool(
                tools,
                link=link,
                slug=(slug if (slug and not link) else None),
                project_hint=project_hint,
                event_end_utc=end_iso or "",
                pre_days=7,
                post_days=7,
            )
            sim = _jaccard(
                cur_tok, _tokens((p.get("title") or "") + " " + (p.get("body") or ""))
            )
            out.append(
                {
                    "id": pid,
                    "title": p.get("title"),
                    "author": p.get("author"),
                    "end_utc": end_iso,
                    "timeline_metrics": timeline,
                    "price_impact_pct": price_imp,
                    "tvl_impact_pct": tvl_imp,
                    "similarity": round(float(sim), 4),
                }
            )
        except Exception as e:
            print(f"[adjacent] failed for {p.get('id')}: {e}")
    return out


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
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
            llm_provider = get_llm_provider()
        except ValueError as exc:
            raise SystemExit(
                "No LLM provider configured. Populate .env or env vars."
            ) from exc

        # one-time semantic refs
        semantic_refs = _bootstrap_semantic_references(all_tools, project_root)

        snapshot_url, focus = _interactive_inputs()

        pid = _resolve_pid_from_url(snapshot_url)
        if not pid:
            raise SystemExit("Could not resolve proposal id from URL.")

        # Meta + RESULT (for event times & ex-post result only)
        meta_js = _fetch_meta_by_id(pid)
        result_js = _fetch_result_by_id(pid)  # may be empty if still open

        title = meta_js.get("title")
        body = meta_js.get("body")
        author = meta_js.get("author")
        choices = meta_js.get("choices") or []
        discussion_url = meta_js.get("discussion")
        start_unix = int(meta_js.get("start") or 0)
        end_unix = int(meta_js.get("end") or 0)
        start_iso = _iso_from_unix(start_unix)
        end_iso = _iso_from_unix(end_unix)

        # Tool plan system removed - actual tool calls performed directly

        # Votes (via MCP) + Timeline metrics
        votes = _get_votes_via_snapshot_mcp(all_tools, pid)
        votes_count = len(votes)
        TIMELINE_METRICS = (
            _analyze_timeline(all_tools, start_unix, end_unix, choices, votes)
            if votes
            else {}
        )

        # Token address (from space strategies/meta) + impacts
        token_address = _extract_token_address_from_meta(meta_js)
        ucid = None
        llamalink = None
        slug = None
        # try registry by space
        space = _space_from_url(snapshot_url)
        reg = _registry_csv_lookup_space_only(registry_csv, space=space)
        if reg:
            ucid = _clean_ucid(reg.get("cmc_ucid"))
            llamalink = reg.get("defillama_link")
            slug = _slug_from_defillama_link(llamalink)
        project_hint = _pick_project_hint(space, title)

        token_price_impact = (
            compute_token_price_impact_from_parquet(
                project_root / "cmc_historical_daily_2013_2025.parquet",
                ucid or "",
                end_iso or "",
            )
            if ucid and end_iso
            else None
        )
        tvl_impact = compute_tvl_impact_from_defillama_tool(
            all_tools,
            link=llamalink,
            slug=slug,
            project_hint=project_hint,
            event_end_utc=end_iso or "",
        )


        # Adjacent proposals (select by time/topic) and analytics + similarity
        all_in_space = _fetch_all_proposals_by_space(space) if space else []
        adj = _select_adjacent_proposals(
            all_in_space,
            start_unix,
            current_author=author,
            current_title=title,
            current_body=body,
        )
        ADJACENT_ANALYTICS = _adjacent_analytics(
            all_tools,
            adj,
            cmc_parquet=project_root / "cmc_historical_daily_2013_2025.parquet",
            ucid=ucid,
            link=llamalink,
            slug=slug,
            project_hint=project_hint,
            current_title=title,
            current_body=body,
            max_items=3,
        )

        # -------- Agent pass: decision (LLM consumes metrics + forum comments) --------
        agent_context = DecisionAgentContext(
            snapshot_url=snapshot_url,
            choices=choices,
            discussion_url=discussion_url,
            event_start_utc=start_iso,
            event_end_utc=end_iso,
            votes_count=votes_count,
            timeline_metrics=TIMELINE_METRICS,
            token_price_impact_pct=token_price_impact,
            tvl_impact_pct=tvl_impact,
            adjacent_analytics=ADJACENT_ANALYTICS,
            focus=focus or None,
        )

        decision_output = run_decision_agent(
            tools=tools_for_agent,
            llm_provider=llm_provider,
            context=agent_context,
        )

        print("\n=== ProposalDecision ===")
        if decision_output.pretty_print:
            print(decision_output.pretty_print)
        else:
            print(_to_json_str(decision_output.decision, indent=2))

        decision = decision_output.decision

        # Coercions
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
        if token_address and not decision.address_of_governance_token:
            decision.address_of_governance_token = token_address

        # Inject computed impacts
        if token_price_impact is not None:
            decision.token_price_impact_pct = token_price_impact
        if tvl_impact is not None:
            decision.tvl_impact_pct = tvl_impact

        # Ex-post winner/margins from RESULT (orchestrator-side only)
        scores = (result_js or {}).get("scores") or []
        scores_total = (result_js or {}).get("scores_total")
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

        agentic_choice = decision.selected_choice_label
        actual_outcome = (
            decision.actual_vote_result.winner_label
            if decision.actual_vote_result
            else None
        )

        def _same(a, b):
            if not a or not b:
                return None
            return (
                "same"
                if (str(a).strip().lower() == str(b).strip().lower())
                else "different"
            )

        match_result = _same(agentic_choice, actual_outcome)

        # Persist (timeline + adjacent analytics + semantic refs)
        payload = {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_url": snapshot_url,
            "focus": focus or None,
            "votes_count": votes_count,
            "timeline_metrics_current": TIMELINE_METRICS,
            "adjacent_analytics": ADJACENT_ANALYTICS,  # includes similarity (4)
            "semantic_references": semantic_refs,  # NEW (2; one-time bootstrap)
            "decision": _to_dict(
                decision
            ),  # includes actual_vote_result (3), ai_final_* (5)
            "agentic_ai_choice": agentic_choice,
            "actual_outcome": actual_outcome,
            "match_result": match_result,  # "same" | "different" | None
        }

        logdir = project_root / "Decision_runs"
        logdir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = f"decision_{ts}"
        (logdir / f"{base}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"\nSaved: {logdir / f'{base}.json'}")


if __name__ == "__main__":
    main()
