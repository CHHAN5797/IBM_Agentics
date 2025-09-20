# -*- coding: utf-8 -*-
"""
On-Chain Activity MCP (BigQuery + optional CMC), Improved

Key improvements:
- Reuse shared Snapshot helpers from `agentics.mcp.snapshot_api` when available,
  with a resilient local GraphQL fallback (retry/backoff).
- Price pipeline prioritized as: external_prices (in args) -> local cache -> CMC Pro (optional).
- Pure, reusable metric functions (gini, windowing, abnormal ratios).
- Safer BigQuery query with clear schema normalization and missing-data handling.
- Unified HTTP session with retries; consistent cache-keying for price windows.

Tools (FastMCP):
  - analyze_onchain(AnalyzeInput) -> dict
  - health() -> dict
"""

from __future__ import annotations

import os
import math
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, Field

try:
    # Prefer canonical FastMCP import if available
    from fastmcp import FastMCP
except Exception:
    from mcp.server.fastmcp import FastMCP

# ------------------------------- Config ---------------------------------

UTC = timezone.utc

SNAPSHOT_GQL   = os.getenv("SNAPSHOT_GQL", "https://hub.snapshot.org/graphql")
CMC_BASE       = os.getenv("CMC_BASE", "https://pro-api.coinmarketcap.com")
CMC_API_KEY    = os.getenv("CMC_API_KEY")  # optional if you inject prices
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")

HTTP_TIMEOUT   = int(os.getenv("ONCHAIN_HTTP_TIMEOUT", "60"))
BASE_SLEEP     = float(os.getenv("ONCHAIN_BASE_SLEEP", "0.35"))
RETRY_MAX      = int(os.getenv("ONCHAIN_MAX_RETRIES", "5"))
BACKOFF_BASE   = float(os.getenv("ONCHAIN_BACKOFF_BASE", "1.7"))
UA             = os.getenv("ONCHAIN_UA", "onchain-activity-mcp/1.2")

# Optional price cache directory (when args.price_cache_dir is None)
DEFAULT_PRICE_CACHE = os.getenv("ONCHAIN_PRICE_CACHE", ".cache/prices")

# --------------------------- HTTP session (shared) ----------------------

def build_session() -> requests.Session:
    """Create a shared HTTP session with retries for 429/5xx and a stable UA."""
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter

    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})

    retry = Retry(
        total=RETRY_MAX,
        read=RETRY_MAX,
        connect=RETRY_MAX,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess

HTTP = build_session()

def _sleep():
    """Polite sleep with a bit of jitter to avoid hammering endpoints."""
    j = 0.10 + 0.25 * (hash(time.time()) % 100) / 100.0
    time.sleep(BASE_SLEEP + j)

# ------------------------------- Utilities ------------------------------

def dt_utc(x) -> datetime:
    """Parse various time representations to timezone-aware UTC datetime."""
    if isinstance(x, datetime):
        return x.astimezone(UTC) if x.tzinfo else x.replace(tzinfo=UTC)
    return pd.to_datetime(str(x), utc=True).to_pydatetime().astimezone(UTC)

def to_iso_date(d: datetime) -> str:
    """Return YYYY-MM-DD for a datetime."""
    return d.astimezone(UTC).date().isoformat()

def sha1_key(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()

# ---------------------- Pure metric helpers (reusable) ------------------

def gini_from_weights(w: np.ndarray) -> float:
    """Gini coefficient for nonnegative weights (voting power)."""
    if w.size == 0:
        return 0.0
    x = np.sort(np.abs(w))
    s = x.sum()
    if s == 0:
        return 0.0
    cumx = np.cumsum(x)
    n = x.size
    return float((n + 1 - 2 * (cumx / s).sum() / n))

def top_decile_share(v: np.ndarray) -> float:
    """Share of voting power held by top 10% voters."""
    if v.size == 0:
        return 0.0
    k = max(1, int(math.ceil(0.1 * len(v))))
    idx = np.argsort(v)[::-1][:k]
    tot = v.sum()
    return float(v[idx].sum() / tot) if tot > 0 else 0.0

def abnormal_ratio(level: Optional[float], base: Optional[float]) -> Optional[float]:
    """(level/base) - 1, with None-safety."""
    if level is None or base is None:
        return None
    if np.isnan(level) or np.isnan(base) or base == 0:
        return None
    return float(level / base - 1.0)

# ------------------------------ Snapshot --------------------------------

class _SnapshotLocal:
    """
    Use shared helpers from `agentics.mcp.snapshot_api` when available.
    Falls back to local GraphQL with retries/backoff preserving same semantics.
    """
    def __init__(self, endpoint=SNAPSHOT_GQL):
        self.endpoint = endpoint
        self.s = HTTP

        self._use_shared = False
        try:
            # Import only when available in PYTHONPATH
            from agentics.mcp.snapshot_api import _fetch_proposal_by_id, _fetch_votes_all  # type: ignore
            self._fetch_proposal_by_id = _fetch_proposal_by_id
            self._fetch_votes_all = _fetch_votes_all
            self._use_shared = True
        except Exception:
            self._use_shared = False  # fallback to local HTTP

    def proposal(self, proposal_id: str) -> dict:
        if self._use_shared:
            return self._fetch_proposal_by_id(proposal_id)  # type: ignore
        q = """
        query($id:String!){
          proposal(id:$id){
            id title body start end created choices scores scores_total space{ id name }
          }
        }"""
        return self._gql(q, {"id": proposal_id}).get("proposal") or {}

    def votes(self, proposal_id: str) -> List[dict]:
        if self._use_shared:
            return self._fetch_votes_all(proposal_id)  # type: ignore
        q = """
        query($id:String!, $first:Int!, $skip:Int!){
          votes(first:$first, skip:$skip, where:{proposal:$id}, orderBy:"vp", orderDirection: desc){
            voter vp created choice
          }
        }"""
        out, skip, first = [], 0, 1000
        while True:
            chunk = self._gql(q, {"id": proposal_id, "first": first, "skip": skip})["votes"]
            out.extend(chunk)
            if len(chunk) < first:
                break
            skip += first
        return out

    def _gql(self, query: str, variables: dict) -> dict:
        retries = 0
        while True:
            _sleep()
            try:
                r = self.s.post(self.endpoint, json={"query": query, "variables": variables}, timeout=HTTP_TIMEOUT)
            except requests.RequestException:
                if retries < RETRY_MAX:
                    time.sleep((BACKOFF_BASE ** retries) + 0.1)
                    retries += 1
                    continue
                raise
            if r.status_code == 200:
                js = r.json()
                if "errors" in js:
                    raise RuntimeError(js["errors"])
                return js["data"]
            if r.status_code in (429, 502, 503, 504) and retries < RETRY_MAX:
                ra = r.headers.get("Retry-After")
                delay = float(ra) if (ra and ra.isdigit()) else (BACKOFF_BASE ** retries)
                time.sleep(delay + 0.1)
                retries += 1
                continue
            r.raise_for_status()

# --------------------------- CMC prices (optional) -----------------------

class CMCPrices:
    """
    Minimal CMC Pro client for daily OHLCV close (fallback only).
    This is used only when `external_prices` and local cache are unavailable.
    """
    def __init__(self, base=CMC_BASE, api_key=CMC_API_KEY):
        if not api_key:
            raise RuntimeError("CMC_API_KEY is required when MCP must fetch prices.")
        self.base = base
        self.session = HTTP
        self.session.headers.update({"X-CMC_PRO_API_KEY": api_key})

    def id_by_contract(self, token_addr: str) -> Optional[int]:
        """Resolve CMC id by Ethereum contract address: /v2/cryptocurrency/info?address=<addr>"""
        u = f"{self.base}/v2/cryptocurrency/info"
        r = self.session.get(u, params={"address": token_addr}, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json().get("data", {})
        info = list(data.values())[0] if data else None
        return int(info["id"]) if info and "id" in info else None  # may be None

    def daily_ohlcv(self, cmc_id: int, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch daily OHLCV close prices between [start_date, end_date] in USD.
        """
        def fetch(path):
            u = f"{self.base}{path}"
            params = {
                "id": cmc_id,
                "time_start": start_date,
                "time_end": end_date,
                "interval": "daily",
                "convert": "USD",
            }
            return self.session.get(u, params=params, timeout=HTTP_TIMEOUT)

        r = fetch("/v2/cryptocurrency/ohlcv/historical")
        if r.status_code != 200:
            r = fetch("/v3/cryptocurrency/ohlcv/historical")
        r.raise_for_status()
        js = r.json()
        quotes = js.get("data", {}).get("quotes") or js.get("data", {}).get("ohlcv") or []
        rows = []
        for q in quotes:
            ts = q.get("time_open") or q.get("time_period_start")
            if ts:
                day = pd.to_datetime(ts, utc=True).date()
                close = q["quote"]["USD"]["close"]
                rows.append({"date": day, "price": float(close)})
        return pd.DataFrame(rows).drop_duplicates("date").sort_values("date")

# ----------------------------- BigQuery layer ----------------------------

class BQOnchain:
    """
    Reads ERC-20 token transfers from BigQuery public dataset:
      bigquery-public-data.crypto_ethereum.token_transfers
    """
    def __init__(self, project_id: str):
        if not project_id:
            raise RuntimeError("Set GCP_PROJECT_ID in environment for BigQuery client.")
        # Import lazily to avoid dependency when not used
        from google.cloud import bigquery  # type: ignore
        self.bigquery = bigquery
        self.client = bigquery.Client(project=project_id)

    def token_transfers_daily(self, token_address: str, start_date: str, end_date: str) -> pd.DataFrame:
        sql = """
        DECLARE start_date DATE DEFAULT @start_date;
        DECLARE end_date   DATE DEFAULT @end_date;
        WITH x AS (
          SELECT
            DATE(block_timestamp) AS dt,
            COUNT(DISTINCT transaction_hash) AS n_tx,
            SUM(value) AS volume_raw
          FROM `bigquery-public-data.crypto_ethereum.token_transfers`
          WHERE LOWER(token_address) = LOWER(@token)
            AND DATE(block_timestamp) BETWEEN start_date AND end_date
          GROUP BY dt
        )
        SELECT dt, n_tx, volume_raw
        FROM x
        ORDER BY dt;
        """
        job = self.client.query(
            sql,
            job_config=self.bigquery.QueryJobConfig(
                query_parameters=[
                    self.bigquery.ScalarQueryParameter("token", "STRING", token_address),
                    self.bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
                    self.bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
                ]
            ),
        )
        df = job.result().to_dataframe(create_bqstorage_client=False)
        df.rename(columns={"dt": "date", "volume_raw": "volume"}, inplace=True)
        # Normalize dtypes
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["n_tx"] = pd.to_numeric(df["n_tx"], errors="coerce").fillna(0).astype(int)
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).astype(float)
        return df.sort_values("date")

# ------------------------------ Price cache ------------------------------

def _price_cache_key(token_addr: str, start_date: str, end_date: str) -> str:
    return sha1_key(f"{token_addr.lower()}|{start_date}|{end_date}")

def load_prices_from_cache(cache_dir: str, token_addr: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    key = _price_cache_key(token_addr, start_date, end_date)
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        with path.open("r") as f:
            js = json.load(f)
        df = pd.DataFrame(js)
        if not df.empty and "date" in df:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    return None

def save_prices_to_cache(cache_dir: str, token_addr: str, start_date: str, end_date: str, df: pd.DataFrame):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = _price_cache_key(token_addr, start_date, end_date)
    path = Path(cache_dir) / f"{key}.json"
    js = df.to_dict(orient="records")
    with path.open("w") as f:
        json.dump(js, f)

# ------------------------------- Schemas --------------------------------

class Window(BaseModel):
    pre_days: int = 30
    post_days: int = 30
    base_from: int = 90
    base_to: int = 60

class AnalyzeInput(BaseModel):
    space: str
    proposal_id: str
    token_address: str
    chain_id: int = 1
    cmc_id: Optional[int] = None
    window: Window = Window()
    external_prices: Optional[List[Dict[str, Any]]] = None
    price_cache_dir: Optional[str] = None  # default to DEFAULT_PRICE_CACHE if None

# ------------------------------ Windows ---------------------------------

def build_windows(created: datetime, w: Window) -> Dict[str, Tuple[datetime, datetime]]:
    """Build base/pre/vote/post/event windows around the proposal creation time."""
    created = dt_utc(created)
    return {
        "base":  (created - timedelta(days=w.base_from), created - timedelta(days=w.base_to)),
        "pre":   (created - timedelta(days=w.pre_days),  created - timedelta(days=1)),
        "vote":  (created, created + timedelta(days=7)),  # heuristic if Snapshot end unknown
        "post":  (created + timedelta(days=1), created + timedelta(days=w.post_days)),
        "event": (created - timedelta(days=w.pre_days), created + timedelta(days=w.post_days)),
    }

# ----------------------- Segment stats / reducers ------------------------

def _slice_date(df: pd.DataFrame, win: Tuple[datetime, datetime]) -> pd.DataFrame:
    return df[(df["date"] >= win[0].date()) & (df["date"] <= win[1].date())].copy()

def segment_stats(df: pd.DataFrame, win: Tuple[datetime, datetime],
                  base_tx: Optional[float], base_vol: Optional[float]) -> Dict[str, Optional[float]]:
    seg = _slice_date(df, win)
    if seg.empty:
        return {"n_tx": 0.0, "volume": 0.0, "n_days": 0, "tx_abn": None, "vol_abn": None}
    n_tx = float(seg["n_tx"].sum())
    vol = float(seg["volume"].sum())
    n_days = int(seg["date"].nunique())
    tx_abn = abnormal_ratio(n_tx / max(1, n_days), base_tx)
    vol_abn = abnormal_ratio(vol / max(1, n_days), base_vol)
    return {"n_tx": n_tx, "volume": vol, "n_days": n_days, "tx_abn": tx_abn, "vol_abn": vol_abn}

# ------------------------------ Core MCP --------------------------------

mcp = FastMCP("OnchainActivityMCP")

@mcp.tool()
def analyze_onchain(args: AnalyzeInput) -> Dict[str, Any]:
    """
    Compute on-chain activity windows and simple concentration/abnormal metrics.
    1) Fetch Snapshot proposal meta & votes (shared helper preferred).
    2) Build time windows around the proposal's creation/start/end.
    3) Load prices (external -> cache -> CMC Pro fallback).
    4) Query BigQuery for daily transfers; compute window stats and abnormal ratios.
    """
    # 1) Snapshot
    snap = _SnapshotLocal()
    meta = snap.proposal(args.proposal_id) or {}
    votes = snap.votes(args.proposal_id) or []
    votes_sorted = sorted(votes, key=lambda v: int(v.get("created") or 0))

    # Robust created time: prefer 'created'; fallback to 'start' if missing
    created_ts = int(meta.get("created") or 0) or int(meta.get("start") or 0)
    created_dt = datetime.fromtimestamp(created_ts, tz=UTC) if created_ts else dt_utc(datetime.utcnow())

    # Vote power vectors for concentration stats
    vp = np.array([float(v.get("vp") or 0.0) for v in votes_sorted], dtype=float)
    gini_vp = gini_from_weights(vp)
    top10_vp = top_decile_share(vp)

    # 2) Windows
    win = build_windows(created_dt, args.window)

    # 3) Prices (optional, not used in metrics here but returned for downstream)
    cache_dir = args.price_cache_dir or DEFAULT_PRICE_CACHE
    prices_df = None
    if args.external_prices:
        df_ext = pd.DataFrame(args.external_prices).copy()
        if not df_ext.empty and {"date","price"}.issubset(df_ext.columns):
            df_ext["date"] = pd.to_datetime(df_ext["date"]).dt.date
            df_ext["price"] = pd.to_numeric(df_ext["price"], errors="coerce")
            prices_df = df_ext.dropna(subset=["date","price"]).drop_duplicates("date").sort_values("date")

    if prices_df is None:
        cached = load_prices_from_cache(cache_dir, args.token_address, to_iso_date(win["event"][0]), to_iso_date(win["event"][1]))
        if cached is not None:
            prices_df = cached

    if prices_df is None and CMC_API_KEY:
        cmc = CMCPrices()
        cmc_id = args.cmc_id or cmc.id_by_contract(args.token_address)
        if cmc_id:
            prices_df = cmc.daily_ohlcv(
                cmc_id,
                to_iso_date(win["event"][0]),
                to_iso_date(win["event"][1]),
            )
            if prices_df is not None and not prices_df.empty:
                save_prices_to_cache(cache_dir, args.token_address, to_iso_date(win["event"][0]), to_iso_date(win["event"][1]), prices_df)

    # 4) BigQuery transfers
    if not GCP_PROJECT_ID:
        raise RuntimeError("GCP_PROJECT_ID is required to query BigQuery for on-chain activity.")
    bq = BQOnchain(GCP_PROJECT_ID)
    transfers = bq.token_transfers_daily(
        args.token_address, to_iso_date(win["event"][0]), to_iso_date(win["event"][1])
    )

    # Base averages
    base_df = _slice_date(transfers, win["base"])
    base_tx = float(base_df["n_tx"].sum() / max(1, base_df["date"].nunique())) if not base_df.empty else None
    base_vol = float(base_df["volume"].sum() / max(1, base_df["date"].nunique())) if not base_df.empty else None

    # Segment stats
    out_stats = {
        "base": segment_stats(transfers, win["base"], base_tx, base_vol),
        "pre":  segment_stats(transfers, win["pre"],  base_tx, base_vol),
        "vote": segment_stats(transfers, win["vote"], base_tx, base_vol),
        "post": segment_stats(transfers, win["post"], base_tx, base_vol),
    }

    return {
        "space": args.space,
        "proposal": {
            "id": meta.get("id"),
            "title": meta.get("title"),
            "created": created_ts,
            "start": meta.get("start"),
            "end": meta.get("end"),
            "choices": meta.get("choices"),
            "scores": meta.get("scores"),
            "scores_total": meta.get("scores_total"),
            "snapshot_space": (meta.get("space") or {}).get("id"),
        },
        "concentration": {
            "gini_vp": round(float(gini_vp), 6),
            "top10_share_vp": round(float(top10_vp), 6),
            "n_votes": int(len(votes_sorted)),
        },
        "windows": {
            k: {"from": int(v[0].timestamp()), "to": int(v[1].timestamp())} for k, v in win.items()
        },
        "onchain": {
            "transfers_daily": transfers.to_dict(orient="records"),
            "stats": out_stats,
        },
        "prices": {
            "source": ("external" if args.external_prices else ("cache" if prices_df is not None and (args.price_cache_dir or DEFAULT_PRICE_CACHE) else ("cmc" if CMC_API_KEY else "none"))),
            "series": (prices_df.to_dict(orient="records") if prices_df is not None else []),
        },
        "notes": {
            "snapshot_via_shared_helpers": True if getattr(snap, "_use_shared", False) else False,
            "cache_dir": cache_dir,
        }
    }

@mcp.tool()
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "OnchainActivityMCP"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
