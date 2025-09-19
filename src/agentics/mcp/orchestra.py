#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAO Governance Orchestrator — Rich Report + No-Lookahead + Efficient IO + Online Learning

- 의사결정(features)은 cutoff 이전 데이터만 사용 (no lookahead)
- 각 제안 결정 이후 label(사후 시장반응)로 온라인 학습 → 다음 제안에 반영
- 항상 동일 스키마로 결과 저장(정규화)
- SHOW_MCP=1 환경변수로 MCP 단계별 start/done JSON 라인 출력
"""

from __future__ import annotations
import os, sys, json, argparse, importlib.util, traceback
from datetime import datetime, timezone, timedelta, date
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    print("This script needs numpy. Please `pip install numpy`.", file=sys.stderr)
    sys.exit(1)

UTC  = timezone.utc
HERE = os.path.dirname(os.path.abspath(__file__))
VERBOSE_MCP = os.environ.get("SHOW_MCP") == "1"

# ------------------------------ Dynamic imports ------------------------------
def _load(path: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {modname} from {path}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m

MODS = {
    "snapshot_api":               os.path.join(HERE, "snapshot_api.py"),
    "timeline_mcp":               os.path.join(HERE, "timeline_mcp.py"),
    "forums_mcp":                 os.path.join(HERE, "forums_mcp.py"),
    "cmc_mcp":                    os.path.join(HERE, "cmc_mcp.py"),
    "defillama_mcp":              os.path.join(HERE, "defillama_mcp.py"),
    "govnews_mcp":                os.path.join(HERE, "govnews_mcp.py"),
    "holders_activity_mcp":       os.path.join(HERE, "holders_activity_mcp.py"),
    "semantics_mcp":              os.path.join(HERE, "semantics_mcp.py"),
    "onchain_activity_mcp_bq_cmc":os.path.join(HERE, "onchain_activity_mcp_bq_cmc.py"),
}

SNP   = _load(MODS["snapshot_api"], "snapshot_api")
TLM   = _load(MODS["timeline_mcp"], "timeline_mcp")
FRM   = _load(MODS["forums_mcp"], "forums_mcp")
CMC   = _load(MODS["cmc_mcp"], "cmc_mcp")
LLAMA = _load(MODS["defillama_mcp"], "defillama_mcp")
NEWS  = _load(MODS["govnews_mcp"], "govnews_mcp")
HOLD  = _load(MODS["holders_activity_mcp"], "holders_activity_mcp")
SEMS  = _load(MODS["semantics_mcp"], "semantics_mcp")
try:
    ONCH = _load(MODS["onchain_activity_mcp_bq_cmc"], "onchain_activity_mcp_bq_cmc")
except Exception:
    ONCH = None

def _mcp_log(what: str, **extra):
    if VERBOSE_MCP:
        rec = {"mcp": what}
        if extra: rec.update(extra)
        print(json.dumps(rec, ensure_ascii=False))

# --------------------------------- Helpers -----------------------------------
def _to_ts(x: int | float | str | datetime) -> int:
    if isinstance(x, datetime): return int(x.astimezone(UTC).timestamp())
    if isinstance(x, (int, float)): return int(x)
    s = str(x).replace("Z", "")
    return int(datetime.fromisoformat(s).replace(tzinfo=UTC).timestamp())

def _to_iso(d: datetime) -> str:
    return d.astimezone(UTC).date().isoformat()

def _event_window(center_ts: int, pre_days: int, post_days: int) -> Tuple[str, str]:
    c = datetime.fromtimestamp(center_ts, tz=UTC)
    a = c - timedelta(days=int(pre_days))
    b = c + timedelta(days=int(post_days))
    return _to_iso(a), _to_iso(b)

import asyncio
import inspect

def _tool_invoke(tool, /, **kwargs):
    """
    Generic invoker that works with:
      - plain callables (sync)
      - callables returning coroutines (async)
      - objects exposing .call/.invoke/.run/.func (sync or async)
    Returns the resolved value (awaited if needed).
    """
    def _maybe_await(x):
        if inspect.iscoroutine(x):
            try:
                # If inside an event loop (unlikely here), create a new loop to avoid "already running" errors
                try:
                    loop = asyncio.get_running_loop()
                    # running loop -> use a dedicated new loop in a nested thread
                    # but simpler for this script: just run asyncio.run (will error if loop is running)
                    # So fall back to a manual loop if needed.
                    # We'll try asyncio.run and if it fails, fall back.
                    try:
                        return asyncio.run(asyncio.shield(x))
                    except RuntimeError:
                        # Fallback: create a new loop
                        new_loop = asyncio.new_event_loop()
                        try:
                            asyncio.set_event_loop(new_loop)
                            return new_loop.run_until_complete(x)
                        finally:
                            new_loop.close()
                            asyncio.set_event_loop(loop)
                except RuntimeError:
                    # No running loop -> safe to asyncio.run
                    return asyncio.run(x)
            except Exception as e:
                # As a last resort: raise with context
                raise
        return x

    # 1) Direct callable
    if callable(tool):
        res = tool(**kwargs)
        return _maybe_await(res)

    # 2) Try common method names on wrappers
    for meth in ("call", "invoke"):
        m = getattr(tool, meth, None)
        if m and callable(m):
            res = m(**kwargs)
            return _maybe_await(res)

    # 3) Some wrappers use .run
    m = getattr(tool, "run", None)
    if m and callable(m):
        try:
            res = m(kwargs)
        except TypeError:
            res = m(**kwargs)
        return _maybe_await(res)

    # 4) Or .func
    m = getattr(tool, "func", None)
    if m and callable(m):
        res = m(**kwargs)
        return _maybe_await(res)

    raise TypeError(f"Unsupported tool type for invocation: {type(tool)}")


POS_WORDS = {"credible","long-term","sustainable","security","risk-reduction",
             "align","transparency","audit","decentralization","efficiency"}
NEG_WORDS = {"rent-seeking","short-term","dilution","centralize","opaque",
             "conflict","risky","attack","exploit","spam"}

def quick_sentiment(texts: List[str]) -> Dict[str, float]:
    if not texts: return {"pos":0.0,"neg":0.0,"score":0.0}
    pos = sum(any(w in t.lower() for w in POS_WORDS) for t in texts)
    neg = sum(any(w in t.lower() for w in NEG_WORDS) for t in texts)
    tot = max(1, len(texts))
    return {"pos":pos/tot, "neg":neg/tot, "score":(pos-neg)/tot}

# ---------------------------- Memoization ----------------------------
from typing import Tuple
_TOKEN_CACHE: Dict[str, Optional[int]] = {}
_SLUG_CACHE: Dict[str, Optional[str]] = {}
_PRICE_SERIES_CACHE: Dict[Tuple[int, str], Dict[str, Any]] = {}
_TVL_SERIES_CACHE: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

# ---------------------------- Fetchers ----------------------------
def fetch_proposals(space: str, start: str, end: str) -> List[dict]:
    _mcp_log("snapshot.list_finished_proposals.start", space=space, start=start, end=end)
    all_props = _tool_invoke(SNP._fetch_all_proposals, space=space)
    finished  = _tool_invoke(SNP._finished_only, proposals=all_props)
    s_ts = _to_ts(start + " 00:00:00+00:00")
    e_ts = _to_ts(end   + " 23:59:59+00:00")
    out = [p for p in finished if s_ts <= int(p.get("end") or 0) <= e_ts]
    out.sort(key=lambda p: int(p.get("end") or 0))  # time order
    _mcp_log("snapshot.list_finished_proposals.done", count=len(out))
    return out

def fetch_votes(proposal_id: str) -> List[dict]:
    """
    Robust wrapper for snapshot_api._fetch_votes_all with multiple signature fallbacks.
    Tries in order:
      1) proposal_id=...
      2) proposal=...
      3) id=...
      4) positional only
    """
    _mcp_log("snapshot.get_votes_all.start", proposal_id=proposal_id)

    fn = getattr(SNP, "_fetch_votes_all", None)
    if fn is None:
        _mcp_log("snapshot.get_votes_all.error", error="_fetch_votes_all not found")
        return []

    # 1) proposal_id=...
    try:
        v = _tool_invoke(fn, proposal_id=proposal_id)
        _mcp_log("snapshot.get_votes_all.done", count=len(v), via="proposal_id")
        return v
    except TypeError:
        pass

    # 2) proposal=...
    try:
        v = _tool_invoke(fn, proposal=proposal_id)
        _mcp_log("snapshot.get_votes_all.done", count=len(v), via="proposal")
        return v
    except TypeError:
        pass

    # 3) id=...
    try:
        v = _tool_invoke(fn, id=proposal_id)
        _mcp_log("snapshot.get_votes_all.done", count=len(v), via="id")
        return v
    except TypeError:
        pass

    # 4) positional only
    try:
        if callable(fn):
            v = fn(proposal_id)
        else:
            # 일부 래퍼형은 kwargs를 무시하고 내부에서 읽기도 함
            v = _tool_invoke(fn, **{})
        _mcp_log("snapshot.get_votes_all.done", count=len(v), via="positional")
        return v
    except Exception as e:
        _mcp_log("snapshot.get_votes_all.error", error=str(e))
        return []

def analyze_timeline_with_cutoff(proposal: dict, votes: List[dict], cutoff_ts: int) -> dict:
    start = int(proposal.get("start") or 0)
    end   = cutoff_ts
    choices = proposal.get("choices") or []
    _mcp_log("timeline.analyze.start", start=start, end=end, choices=len(choices), votes=len(votes))
    out = _tool_invoke(TLM.analyze_timeline, start=start, end=end, choices=choices, votes=votes)
    _mcp_log("timeline.analyze.done")
    return out

def fetch_forum_summary(url: Optional[str], cutoff_ts: Optional[int]=None) -> Dict[str, Any]:
    if not url:
        return {"summary":None,"sentiment":{"pos":0,"neg":0,"score":0},"posts_count":0}
    _mcp_log("forums.fetch.start", url=url)
    disc = _tool_invoke(FRM.fetch_discussion, url=url, max_pages=3)
    posts_raw = (disc.get("posts") or [])
    filtered_posts = []
    if cutoff_ts is not None:
        for p in posts_raw:
            ts = None
            if "created_at" in p and p["created_at"]:
                try: ts = _to_ts(p["created_at"])
                except Exception: ts = None
            elif "updated_at" in p and p["updated_at"]:
                try: ts = _to_ts(p["updated_at"])
                except Exception: ts = None
            if ts is None or ts <= cutoff_ts:
                filtered_posts.append(p)
    else:
        filtered_posts = posts_raw
    texts = [p.get("raw") or p.get("cooked") or "" for p in filtered_posts]
    sent = quick_sentiment(texts)
    _mcp_log("forums.fetch.done", kept=len(filtered_posts))
    return {"header":disc.get("header"), "posts_count":len(filtered_posts), "sentiment":sent}

def resolve_price_token(project_hint: str) -> Optional[int]:
    if project_hint in _TOKEN_CACHE: return _TOKEN_CACHE[project_hint]
    try:
        _mcp_log("cmc.resolve_tokens.start", project=project_hint)
        res = _tool_invoke(CMC.resolve_tokens, project_hint=project_hint, prefer_governance=True)
        cid = res.get("best", {}).get("id") or (res.get("candidates") or [{}])[0].get("id")
        out = int(cid) if cid else None
    except Exception:
        out = None
    _TOKEN_CACHE[project_hint] = out
    _mcp_log("cmc.resolve_tokens.done", cmc_id=out)
    return out

def price_window_batched(cmc_id: int, start_date: str, end_date: str, interval: str="1d") -> List[dict]:
    key = (cmc_id, interval)
    cur = _PRICE_SERIES_CACHE.get(key)
    if not cur:
        _mcp_log("cmc.price_window.start", cmc_id=cmc_id, start=start_date, end=end_date, interval=interval)
        rows = (_tool_invoke(CMC.price_window, token=cmc_id, interval=interval,
                             start_date=start_date, end_date=end_date) or {}).get("rows") or []
        _PRICE_SERIES_CACHE[key] = {"rows": rows, "start": start_date, "end": end_date}
        _mcp_log("cmc.price_window.done", got=len(rows))
        return rows
    if start_date < cur["start"]:
        rows = (_tool_invoke(CMC.price_window, token=cmc_id, interval=interval,
                             start_date=start_date, end_date=cur["start"]) or {}).get("rows") or []
        cur["rows"] = (rows + cur["rows"]); cur["start"] = start_date
    if end_date > cur["end"]:
        rows = (_tool_invoke(CMC.price_window, token=cmc_id, interval=interval,
                             start_date=cur["end"], end_date=end_date) or {}).get("rows") or []
        cur["rows"] = (cur["rows"] + rows); cur["end"] = end_date
    return cur["rows"]

def llama_slug_from_space(space: str) -> Optional[str]:
    if space in _SLUG_CACHE: return _SLUG_CACHE[space]
    try:
        _mcp_log("defillama.refresh_protocols_cache.start")
        _tool_invoke(LLAMA.refresh_protocols_cache, ttl_hours=24)
        _mcp_log("defillama.refresh_protocols_cache.done")
        out = _tool_invoke(LLAMA.guess_slug_by_space, space=space)
    except Exception:
        out = None
    _SLUG_CACHE[space] = out
    return out

def tvl_window_batched(slug: str, event_ts: int, pre: int, post: int) -> List[dict]:
    key = (slug, pre, post)
    cur = _TVL_SERIES_CACHE.get(key)
    if not cur:
        center_iso = datetime.fromtimestamp(event_ts, tz=UTC).isoformat()
        _mcp_log("defillama.event_window.start", slug=slug, center=center_iso, pre=pre, post=post)
        rows = (_tool_invoke(LLAMA.event_window, slug=slug, event_time_utc=center_iso,
                             pre_days=pre, post_days=post) or {}).get("rows") or []
        _TVL_SERIES_CACHE[key] = {"rows": rows, "min_ts": event_ts - pre*86400, "max_ts": event_ts + post*86400}
        _mcp_log("defillama.event_window.done", got=len(rows))
        return rows
    return cur["rows"]

def news_window(project_hint: str, title_or_id: str, event_ts: int, pre: int, post: int) -> List[dict]:
    event_iso = datetime.fromtimestamp(event_ts, tz=UTC).isoformat()
    _mcp_log("govnews.window.start", project=project_hint, title=title_or_id)
    out = _tool_invoke(NEWS.proposal_news_window,
                       project_hint=project_hint,
                       proposal_title_or_id=title_or_id,
                       event_time_utc=event_iso,
                       pre_days=pre, post_days=post,
                       lang="en", max_records=60,
                       ttl_minutes=int(os.environ.get("NEWS_TTL_MINUTES","30")))
    _mcp_log("govnews.window.done")
    return out.get("articles") or []

GOV_TOKEN_ADDR = {
    "aavedao.eth": "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",  # AAVE
}
def holders_summary(token_address: Optional[str]) -> Optional[dict]:
    if not token_address: return None
    _mcp_log("holders.analyze.start", token=token_address)
    try:
        out = _tool_invoke(HOLD.analyze_holders, token_address=token_address, top_k=100)
        _mcp_log("holders.analyze.done")
        return out
    except Exception as e:
        _mcp_log("holders.analyze.error", error=str(e))
        return None

# ---------------------- Features & labels ----------------------
def _safe_get(d: dict, path: List[str], default=None):
    cur = d or {}
    for k in path:
        if not isinstance(cur, dict): return default
        cur = cur.get(k)
    return cur if cur is not None else default

def build_features(timeline: dict, forum: dict) -> List[float]:
    early = _safe_get(timeline, ["early_lead_hits"], {}) or {}
    early_ratio = (1.0 * sum(1 for q in ("Q1","Q2","Q3","Q4") if early.get(q))) / 4.0 if early else 0.0
    spike_idx   = _safe_get(timeline, ["spike","spike_index"], 0.0) or 0.0
    forum_score = _safe_get(forum, ["sentiment","score"], 0.0) or 0.0
    return [early_ratio, spike_idx, forum_score]

def _pct_change(a: float, b: float) -> Optional[float]:
    try:
        if a is None or b is None or b == 0: return None
        return (a - b) / abs(b)
    except Exception:
        return None

def _series_pct_change(rows: List[dict], key: str, t0: str, t1: str) -> Optional[float]:
    if not rows: return None
    pre_vals  = [r.get(key) for r in rows if r.get("date") <= t0 and r.get(key) is not None]
    post_vals = [r.get(key) for r in rows if r.get("date") >= t1 and r.get(key) is not None]
    if not pre_vals or not post_vals: return None
    pre  = float(pre_vals[-1]); post = float(post_vals[0])
    return _pct_change(post, pre)

def compute_market_label(end_ts: int, pre: int, post: int,
                         prices: List[dict], tvls: List[dict]) -> Optional[int]:
    pre_cut  = (datetime.fromtimestamp(end_ts, tz=UTC) - timedelta(days=1)).date().isoformat()
    post_cut = (datetime.fromtimestamp(end_ts, tz=UTC) + timedelta(days=1)).date().isoformat()
    price_delta = _series_pct_change(prices, "price_usd", pre_cut, post_cut)
    tvl_delta   = _series_pct_change(tvls, "tvl_usd",   pre_cut, post_cut)
    if price_delta is None and tvl_delta is None: return None
    comp = (0.6 * (price_delta or 0.0)) + (0.4 * (tvl_delta or 0.0))
    return 1 if comp > 0 else 0

# ----------------------------- Event Study / DiD -----------------------------
def _to_returns(rows: List[dict], key: str="price_usd") -> List[Tuple[str, float]]:
    xs = []; prev = None
    for r in sorted(rows, key=lambda z: z.get("date")):
        v = r.get(key)
        if v is None: continue
        v = float(v)
        if prev is None:
            prev = v; continue
        ret = (v - prev) / (prev if prev != 0 else 1.0)
        xs.append((r.get("date"), float(ret)))
        prev = v
    return xs

def _ols_alpha_beta(ret_i: List[float], ret_m: List[float]) -> Tuple[float, float]:
    x = np.array(ret_m, dtype=float); y = np.array(ret_i, dtype=float)
    if len(x) < 3 or len(y) != len(x): return 0.0, 0.0
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.pinv(X).dot(y)
    return float(beta[0]), float(beta[1])

def compute_car_market_model(price_rows: List[dict], end_ts: int, pre: int, post: int,
                             lookback: int = 60, market_rows: Optional[List[dict]] = None) -> Optional[dict]:
    if not price_rows: return None
    rets_i = _to_returns(price_rows)
    if len(rets_i) < (lookback + pre + post + 5): return None
    if market_rows:
        rets_m = _to_returns(market_rows)
        idx = {d: r for d, r in rets_m}
        aligned_m = [idx.get(d, 0.0) for d, _ in rets_i]
    else:
        aligned_m = [0.0 for _ in rets_i]
    dates = [d for d, _ in rets_i]
    event_day = datetime.fromtimestamp(end_ts, tz=UTC).date().isoformat()
    try:
        t_event = dates.index(event_day)
    except ValueError:
        def _dist(di):
            dd = datetime.fromisoformat(dates[di]).date()
            return abs((dd - datetime.fromtimestamp(end_ts, tz=UTC).date()).days)
        t_event = min(range(len(dates)), key=_dist)

    t_est_start = max(0, t_event - pre - lookback)
    t_est_end   = max(0, t_event - pre - 1)
    est_i = [r for _, r in rets_i[t_est_start:t_est_end+1]]
    est_m = aligned_m[t_est_start:t_est_end+1]
    a, b = _ols_alpha_beta(est_i, est_m)

    t1 = max(0, t_event - 1)
    t2 = min(len(rets_i)-1, t_event + 1)
    AR = []
    for t in range(t1, t2+1):
        ri = rets_i[t][1]; rm = aligned_m[t]
        exp = a + b * rm
        AR.append(ri - exp)
    CAR = float(np.sum(AR))
    var = float(np.var(AR)) if len(AR) > 1 else 0.0
    return {"window": [-1, +1], "alpha": a, "beta": b, "CAR": CAR, "var": var, "AR": AR}

def compute_did_tvl(tvl_rows_proto: List[dict], tvl_rows_controls: List[List[dict]],
                    end_ts: int, pre: int, post: int) -> Optional[dict]:
    if not tvl_rows_proto or not tvl_rows_controls: return None
    def _first_after(rows, day):
        vals = [r["tvl_usd"] for r in sorted(rows, key=lambda z:z["date"]) if r.get("date") >= day and r.get("tvl_usd") is not None]
        return vals[0] if vals else None
    def _last_before(rows, day):
        vals = [r["tvl_usd"] for r in sorted(rows, key=lambda z:z["date"]) if r.get("date") <= day and r.get("tvl_usd") is not None]
        return vals[-1] if vals else None

    pre_cut  = (datetime.fromtimestamp(end_ts, tz=UTC) - timedelta(days=1)).date().isoformat()
    post_cut = (datetime.fromtimestamp(end_ts, tz=UTC) + timedelta(days=1)).date().isoformat()

    p0 = _last_before(tvl_rows_proto, pre_cut)
    p1 = _first_after(tvl_rows_proto, post_cut)
    if p0 is None or p1 is None: return None
    d_proto = (p1 - p0)

    ds_ctrl = []
    for rows in tvl_rows_controls:
        c0 = _last_before(rows, pre_cut)
        c1 = _first_after(rows, post_cut)
        if c0 is not None and c1 is not None:
            ds_ctrl.append(c1 - c0)
    if not ds_ctrl: return None
    did = d_proto - float(np.mean(ds_ctrl))
    return {"DiD": float(did), "controls_n": int(len(ds_ctrl))}

# ----------------------------- Minimal Logistic (online) ---------------------
class LogisticModel:
    def __init__(self, n_features: int, lr: float=0.1, l2: float=0.0, max_iter: int=200):
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
        self.max_iter = max_iter
    @staticmethod
    def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
    def set_params(self, w: np.ndarray, b: float):
        self.w = np.array(w, dtype=float).copy(); self.b = float(b)
    def fit(self, X: np.ndarray, y: np.ndarray, warm_start: bool=False):
        n, d = X.shape; w, b = self.w, self.b
        for _ in range(self.max_iter):
            z = X.dot(w) + b
            p = self._sigmoid(z)
            grad_w = (X.T.dot(p - y) / n) + self.l2 * w
            grad_b = np.mean(p - y)
            w -= self.lr * grad_w; b -= self.lr * grad_b
        self.w, self.b = w, b
    def partial_fit(self, x: np.ndarray, y: float, steps: int = 80):
        w, b = self.w, self.b
        for _ in range(steps):
            z = float(x.dot(w) + b)
            p = self._sigmoid(z)
            err = (p - y)
            grad_w = err * x + self.l2 * w
            grad_b = err
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.w, self.b = w, b
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X.dot(self.w) + self.b)

# ------------------------------ Agent policies -------------------------------
def advisor_decide_rule_based(timeline: dict, forum_sent: Dict[str, float]) -> Tuple[str, str]:
    rec_idx = timeline.get("recommended_index")
    if rec_idx is None:
        return "ABSTAIN", "No clear timeline signal before cutoff."
    spike = (timeline.get("spike", {}) or {}).get("spike_index", 0.0)
    sscore = (forum_sent or {}).get("score", 0.0)
    if spike > 0.40 and sscore < 0:
        return "ABSTAIN", f"Large late spike ({spike:.2f}) with negative forum tone ({sscore:.2f})."
    return ("YES", f"Stable early lead (timeline rec={rec_idx}); forum tone {sscore:+.2f}.")

def advisor_decide_learned(proba_positive: float, threshold_yes: float=0.55, abstain_band: float=0.05) -> Tuple[str,str]:
    if proba_positive >= threshold_yes:
        return "YES", f"Learned policy: p(positive)={proba_positive:.2f} >= {threshold_yes:.2f}"
    if proba_positive <= (threshold_yes - abstain_band):
        return "NO",  f"Learned policy: p(positive)={proba_positive:.2f} <= {threshold_yes-abstain_band:.2f}"
    return "ABSTAIN", f"Learned policy: p(positive)={proba_positive:.2f} (indeterminate band)"

# ------------------------- Semantics helper -----------------------
def semantics_candidates_with_year_cap(query: str, year_cap: int, k: int=8) -> List[dict]:
    try:
        _mcp_log("semantics.search.start", q=query, year_cap=year_cap)
        cands = _tool_invoke(SEMS.s2_search, query=query, k=k)
    except Exception:
        cands = []
    filtered = []
    for p in cands or []:
        y = p.get("year") if isinstance(p, dict) else None
        try:
            if y is None or int(y) <= int(year_cap):
                filtered.append(p)
        except Exception:
            filtered.append(p)
    norm = getattr(SEMS, "_normalize_paper", None)
    out = [norm(x) for x in filtered[:k]] if callable(norm) else filtered[:k]
    _mcp_log("semantics.search.done", kept=len(out))
    return out

# ------------------------ Normalizer (schema consistency) --------------------
def _normalize_row(row: dict) -> dict:
    base = {
        "proposal": {"id": None, "title": None, "discussion": None, "end_ts": None, "choices": None},
        "cutoff_ts": None,
        "features": [0.0, 0.0, 0.0],
        "variables": {"early_lead_ratio": 0.0, "spike_index": 0.0, "forum_sentiment_score": 0.0},
        "timeline": {},
        "forum": {"posts_count": 0, "sentiment": {"pos": 0.0, "neg": 0.0, "score": 0.0}},
        "natural_result": {},
        "prices": [],
        "tvls": [],
        "news": [],
        "holders": None,
        "metrics": {"car_market_model": None, "tvl_did": None},
        "label": None,
        "semantics": {"goal": None, "plan_prompt": None, "query": None, "sources": []},
        "simulation_steps": [],
        "agent_decision": {"vote": "ABSTAIN", "rationale": "insufficient data", "p_positive": None},
        "why_decision": "Fallback due to missing pieces."
    }
    def merge(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            out = dict(a)
            for k, v in b.items():
                out[k] = merge(a.get(k), v)
            return out
        return b if b is not None else a
    return merge(base, row)

# ------------------------ Per-proposal analysis ------------------------------
def analyze_one(space: str, proposal: dict, price_interval: str, pre: int, post: int,
                decide_hours_before_end: int, goal: str) -> dict:
    pid   = proposal["id"]
    title = proposal.get("title") or pid
    discussion = proposal.get("discussion")
    end_ts = int(proposal.get("end") or 0)
    cutoff_ts = end_ts - max(0, int(decide_hours_before_end)) * 3600

    # 1) votes up to cutoff
    all_votes = fetch_votes(pid)
    votes = [v for v in all_votes if int(v.get("created") or 0) <= cutoff_ts]

    # 2) timeline & forum (cutoff-safe)
    tline = analyze_timeline_with_cutoff(proposal, votes, cutoff_ts=cutoff_ts)
    forum = fetch_forum_summary(discussion, cutoff_ts=cutoff_ts)

    # 3) evaluation-only (not for decision)
    project_hint = space.split(".")[0]
    cmc_id = resolve_price_token(project_hint)
    start_iso, end_iso = _event_window(end_ts, pre, post)
    prices = price_window_batched(cmc_id, start_iso, end_iso, interval=price_interval) if cmc_id else []
    slug  = llama_slug_from_space(space)
    tvls  = tvl_window_batched(slug, end_ts, pre, post) if slug else []
    articles = news_window(project_hint, title, end_ts, pre, post)
    token_addr = GOV_TOKEN_ADDR.get(space)
    holders = holders_summary(token_addr) if token_addr else None
    nat = _tool_invoke(SNP._fetch_proposal_result_by_id, proposal_id=pid)

    # 4) semantics (<= cutoff year)
    cutoff_year = datetime.fromtimestamp(cutoff_ts, tz=UTC).year if cutoff_ts else datetime.fromtimestamp(end_ts, tz=UTC).year
    plan_prompt = (
        f"Goal: {goal} for {space}. "
        f"Use timeline/forum up to cutoff {datetime.fromtimestamp(cutoff_ts, tz=UTC).isoformat()}Z; "
        f"do NOT use final tally. Cite methods up to {cutoff_year}."
    )
    sem_q = f"{project_hint} governance proposal event study + DeFi TVL price response methodology"
    cands = semantics_candidates_with_year_cap(sem_q, cutoff_year, k=8)
    refs = []
    for p in cands or []:
        refs.append({
            "title": p.get("title"), "year": p.get("year"),
            "authors": p.get("authors"), "venue": p.get("venue"),
            "doi": p.get("doi"), "url": p.get("url")
        })

    # 5) compute features/labels/metrics
    feats = build_features(tline, forum)
    car = compute_car_market_model(prices, end_ts=end_ts, pre=pre, post=post, lookback=60, market_rows=None)
    did = None
    label = compute_market_label(end_ts, pre, post, prices, tvls)

    variables_desc = {"early_lead_ratio": feats[0], "spike_index": feats[1], "forum_sentiment_score": feats[2]}
    simulation_steps = [
        "Fetched snapshot proposal and votes up to cutoff (no lookahead)",
        "Analyzed timeline stability/lead and forum tone before cutoff",
        "Resolved token/protocol ids; pulled price/TVL windows for evaluation",
        "Computed CAR (market model) and TVL DiD as market response metrics",
        "Searched semantic literature for governance/event-study methods (<= cutoff year)",
        "Formed agent decision using rule-based or learned policy"
    ]

    return {
        "proposal": {"id": pid, "title": title, "discussion": discussion,
                     "end_ts": end_ts, "choices": proposal.get("choices")},
        "cutoff_ts": cutoff_ts,
        "features": feats,
        "variables": variables_desc,
        "timeline": tline,
        "forum": forum,
        "natural_result": nat,
        "prices": prices,
        "tvls": tvls,
        "news": articles,
        "holders": holders,
        "metrics": {"car_market_model": car, "tvl_did": did},
        "label": label,
        "semantics": {"goal": goal, "plan_prompt": plan_prompt, "query": sem_q, "sources": refs},
        "simulation_steps": simulation_steps
    }

# ------------------------------ Model IO utils -------------------------------
def save_model(path: str, model: LogisticModel, trained_until: str,
               feature_names: List[str], trained_ids: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path,
             w=model.w, b=model.b,
             trained_until=trained_until,
             feature_names=np.array(feature_names, dtype=object),
             trained_ids=np.array(trained_ids, dtype=object))

def load_model(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    res = {"w": d["w"], "b": float(d["b"]),
           "trained_until": str(d["trained_until"]),
           "feature_names": list(d["feature_names"].tolist())}
    if "trained_ids" in d:
        res["trained_ids"] = set([str(x) for x in d["trained_ids"].tolist()])
    else:
        res["trained_ids"] = set()
    return res

def date_str_to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

# ---------------------------------- Main -------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--train-start")
    ap.add_argument("--train-end")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end",   required=True)
    ap.add_argument("--price-interval", default="1d", choices=("1d","1h"))
    ap.add_argument("--pre", type=int, default=7)
    ap.add_argument("--post", type=int, default=7)
    ap.add_argument("--decide-hours-before-end", type=int, default=0)

    ap.add_argument("--model-path")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--load-model", action="store_true")
    ap.add_argument("--train-extend", action="store_true")
    ap.add_argument("--retrain-if-older-than-days", type=int, default=0)
    ap.add_argument("--online-learn", action="store_true", default=True)

    ap.add_argument("--yes-threshold", type=float, default=0.55)
    ap.add_argument("--abstain-band", type=float, default=0.05)
    ap.add_argument("--goal", default="Maximize long-term DAO growth and community welfare via governance.")

    args = ap.parse_args()

    data_dir = os.path.abspath(os.environ.get("DATA_DIR", "./data"))
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "errors"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "models"), exist_ok=True)

    feature_names = ["early_lead_ratio","spike_index","forum_sentiment_score"]

    # Load existing model
    loaded = None
    trained_ids: set[str] = set()
    if args.load_model and args.model_path:
        loaded = load_model(args.model_path)
        if loaded:
            trained_ids = loaded.get("trained_ids", set())
            print(json.dumps({"model_loaded": True, "model_path": args.model_path,
                              "trained_until": loaded["trained_until"],
                              "trained_ids_count": len(trained_ids)}, ensure_ascii=False))
        else:
            print(json.dumps({"model_loaded": False, "reason": "file_missing"}, ensure_ascii=False))

    # Optional offline training
    model = None
    trained_until_str = None
    infer_start_ts = _to_ts(args.start + " 00:00:00+00:00")
    if args.train_start and args.train_end:
        train_props_all = fetch_proposals(args.space, args.train_start, args.train_end)
        train_props = [p for p in train_props_all if int(p.get("end") or 0) <= infer_start_ts]

        if loaded and args.train_extend and loaded.get("trained_until"):
            trained_until_ts = _to_ts(str(loaded["trained_until"]) + " 00:00:00+00:00")
            train_props = [p for p in train_props if int(p.get("end") or 0) > trained_until_ts]

        X_list, y_list = [], []
        last_end = None
        for p in train_props:
            pid = p.get("id")
            try:
                row = analyze_one(args.space, p, args.price_interval, args.pre, args.post,
                                  decide_hours_before_end=args.decide_hours_before_end, goal=args.goal)
                if row["label"] is None: continue
                X_list.append(row["features"]); y_list.append(int(row["label"]))
                trained_ids.add(str(pid))
                last_end = max(last_end or 0, int(p.get("end") or 0))
            except Exception as e:
                err_path = os.path.join(data_dir, "errors", f"{args.space}_train_{pid}.json")
                with open(err_path, "w", encoding="utf-8") as f:
                    json.dump({"error": str(e), "traceback": traceback.format_exc(), "proposal": p},
                              f, ensure_ascii=False, indent=2)

        if X_list:
            X = np.array(X_list, dtype=float); y = np.array(y_list, dtype=float)
            model = LogisticModel(n_features=X.shape[1], lr=0.2, l2=0.01, max_iter=400)
            if loaded and isinstance(loaded.get("w"), np.ndarray) and loaded["w"].shape[0] == X.shape[1]:
                model.set_params(loaded["w"], loaded["b"])
            model.fit(X, y, warm_start=bool(loaded))
            trained_until_str = datetime.fromtimestamp(last_end or infer_start_ts, tz=UTC).date().isoformat()
            print(json.dumps({
                "train_space": args.space, "train_count": int(X.shape[0]),
                "train_start": args.train_start, "train_end": args.train_end,
                "trained_until": trained_until_str, "features": feature_names
            }, ensure_ascii=False, indent=2))
            if args.save_model and args.model_path:
                save_model(args.model_path, model, trained_until_str, feature_names, list(trained_ids))
                print(json.dumps({"model_saved": True, "model_path": args.model_path}, ensure_ascii=False))
        else:
            if loaded and "w" in loaded:
                model = LogisticModel(n_features=len(feature_names), lr=0.2, l2=0.01, max_iter=1)
                model.set_params(loaded["w"], loaded["b"])
                trained_until_str = loaded["trained_until"]
            print(json.dumps({"training": "online"}, ensure_ascii=False))
    else:
        if loaded and "w" in loaded:
            model = LogisticModel(n_features=len(feature_names), lr=0.2, l2=0.01, max_iter=1)
            model.set_params(loaded["w"], loaded["b"])
            trained_until_str = loaded["trained_until"]
        print(json.dumps({"training": "online"}, ensure_ascii=False))

    # Inference + Online learning
    infer_props = fetch_proposals(args.space, args.start, args.end)
    header = {
        "space": args.space, "count": len(infer_props),
        "start": args.start, "end": args.end,
        "decide_hours_before_end": args.decide_hours_before_end,
        "policy": "learned" if model is not None else "rule-based",
        "model_trained_until": trained_until_str,
        "goal": args.goal,
        "online_learn": bool(args.online_learn)
    }
    print(json.dumps(header, ensure_ascii=False, indent=2))

    out_rows = []
    max_trained_end = _to_ts(trained_until_str + " 00:00:00+00:00") if trained_until_str else None

    for p in infer_props:
        pid = str(p.get("id"))
        try:
            row = analyze_one(args.space, p, args.price_interval, args.pre, args.post,
                              decide_hours_before_end=args.decide_hours_before_end, goal=args.goal)

            # Decide (no lookahead)
            if model is not None and row.get("features"):
                Xq = np.array(row["features"], dtype=float)[None, :]
                proba = float(model.predict_proba(Xq)[0])
                vote, rationale = advisor_decide_learned(proba,
                                                         threshold_yes=args.yes_threshold,
                                                         abstain_band=args.abstain_band)
                row["agent_decision"] = {"vote": vote, "rationale": rationale, "p_positive": proba}
                row["why_decision"] = (
                    "Learned policy used features up to cutoff (no lookahead). "
                    f"Probability of positive market response={proba:.2f}; "
                    f"threshold={args.yes_threshold:.2f}, abstain_band={args.abstain_band:.2f}."
                )
            else:
                vote, rationale = advisor_decide_rule_based(
                    row.get("timeline") or {}, (row.get("forum") or {}).get("sentiment") or {}
                )
                row["agent_decision"] = {"vote": vote, "rationale": rationale, "p_positive": None}
                row["why_decision"] = (
                    "Rule-based policy used timeline lead/spike and forum tone before cutoff; "
                    "no learned model was available at decision time."
                )

            # Normalize schema before appending
            row = _normalize_row(row)
            out_rows.append(row)
            print(json.dumps({"proposal_id": row["proposal"]["id"],
                              "agent_vote": row["agent_decision"]["vote"],
                              "policy": "learned" if model is not None else "rule-based"},
                              ensure_ascii=False))

            # Online update (after decision)
            if args.online_learn and row.get("label") is not None:
                if model is None:
                    model = LogisticModel(n_features=len(row["features"]), lr=0.2, l2=0.01, max_iter=200)
                if pid not in trained_ids:
                    x = np.array(row["features"], dtype=float)
                    y = float(row["label"])
                    model.partial_fit(x, y, steps=80)
                    trained_ids.add(pid)
                    end_ts = int(row["proposal"]["end_ts"])
                    if max_trained_end is None or end_ts > max_trained_end:
                        max_trained_end = end_ts
                        trained_until_str = datetime.fromtimestamp(max_trained_end, tz=UTC).date().isoformat()
                    if args.save_model and args.model_path:
                        save_model(args.model_path, model, trained_until_str or args.start,
                                   feature_names, list(trained_ids))
                        print(json.dumps({"model_saved_online": True,
                                          "trained_ids_count": len(trained_ids),
                                          "trained_until": trained_until_str}, ensure_ascii=False))
        except Exception as e:
            err_path = os.path.join(data_dir, "errors", f"{args.space}_{pid}.json")
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump({"error": str(e), "traceback": traceback.format_exc(), "proposal": p},
                          f, ensure_ascii=False, indent=2)
            print(json.dumps({"error": str(e), "proposal_id": pid}, ensure_ascii=False))
            out_rows.append(_normalize_row({"proposal": {"id": pid, "title": p.get("title")},
                                            "status": "error", "error": str(e)}))

    bundle = {
        "space": args.space,
        "period": {"start": args.start, "end": args.end},
        "goal": args.goal,
        "policy": "learned" if model is not None else "rule-based",
        "model_trained_until": trained_until_str,
        "online_learn": bool(args.online_learn),
        "results": out_rows
    }
    print(json.dumps(bundle, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
