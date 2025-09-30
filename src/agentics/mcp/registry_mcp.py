# src/agentics/mcp/registry_mcp.py
from __future__ import annotations

"""
Registry MCP
- Read DAO registry CSV and resolve governance/native token info for a Snapshot space.
- CSV columns (case-insensitive supported):
  space, CMC_ucid, CMC_ticker, Name_Defillama, Contracts, ...
- Output includes governance_token.address, native_token.ticker, cmc_ucid, defillama name/slug.

ENV:
  DAO_REGISTRY_CSV=absolute/path/to/src/agentics/assets_registry/dao_registry.csv
"""

import os
import re
import csv
from typing import Any, Dict, Optional, List, Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("RegistryMCP")

CSV_PATH = os.getenv(
    "DAO_REGISTRY_CSV",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets_registry", "dao_registry.csv")),
)

def _slugify(name: str) -> str:
    s = (name or "").strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-zA-Z0-9\-]", "", s)
    return s.lower()

def _norm(s: Optional[str]) -> str:
    return (s or "").strip()

def _load_rows() -> List[Dict[str, Any]]:
    if not os.path.exists(CSV_PATH):
        raise RuntimeError(f"DAO registry CSV not found: {CSV_PATH}")
    with open(CSV_PATH, "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        raw = list(reader)

    # normalize keys (lowercase) and map common variants
    normed = []
    for r in raw:
        low = { (k or "").strip().lower(): (v if v is not None else "") for k, v in r.items() }
        row: Dict[str, Any] = {}
        # canonical columns
        row["space"] = _norm(low.get("space"))
        row["cmc_ucid"] = _norm(low.get("cmc_ucid") or low.get("ucid") or low.get("cmc id") or "")
        row["cmc_ticker"] = _norm(low.get("cmc_ticker") or low.get("ticker") or "")
        row["defillama_name"] = _norm(low.get("name_defillama") or low.get("defillama_name") or low.get("name") or "")
        row["contracts"] = _norm(low.get("contracts") or low.get("contract") or "")
        row["cmc_link"] = _norm(low.get("cmc_link") or low.get("cmc url") or "")
        normed.append(row)
    return normed

def _match_row(rows: List[Dict[str, Any]], *, space: Optional[str]=None,
               name: Optional[str]=None, ticker: Optional[str]=None, contract: Optional[str]=None) -> Optional[Dict[str, Any]]:
    if contract:
        c = contract.lower()
        for r in rows:
            contracts = [x.strip().lower() for x in (r["contracts"].split("|") if r["contracts"] else [])]
            if any(x == c for x in contracts):
                return r
    if space:
        s = space.strip().lower()
        for r in rows:
            if (r["space"] or "").strip().lower() == s:
                return r
    if name:
        n = name.strip().lower()
        for r in rows:
            if (r["defillama_name"] or "").strip().lower() == n:
                return r
    if ticker:
        t = ticker.strip().lower()
        for r in rows:
            if (r["cmc_ticker"] or "").strip().lower() == t:
                return r
    return None

class LookupIn(BaseModel):
    space: Optional[str] = Field(None, description="Snapshot space id, e.g., 'aavedao.eth'")
    name: Optional[str] = Field(None, description="DefiLlama protocol name (Name_Defillama)")
    ticker: Optional[str] = Field(None, description="CMC ticker symbol, e.g., 'AAVE'")
    contract: Optional[str] = Field(None, description="Token contract address (0x...)")
    # Optionally compute slug
    want_slug: bool = Field(True, description="Whether to include defillama_slug field")

@mcp.tool(
    name="lookup",
    title="DAO Registry Lookup",
    description="Resolve DAO registry entry by space, name, ticker, or contract address. Returns governance token information, CMC data, and DeFiLlama metadata for DAOs and protocols.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
def lookup(
    space: Annotated[
        Optional[str],
        Field(None, description="Snapshot space id, e.g., 'aavedao.eth'")
    ] = None,
    name: Annotated[
        Optional[str],
        Field(None, description="DefiLlama protocol name (Name_Defillama)")
    ] = None,
    ticker: Annotated[
        Optional[str],
        Field(None, description="CMC ticker symbol, e.g., 'AAVE'")
    ] = None,
    contract: Annotated[
        Optional[str],
        Field(None, description="Token contract address (0x...)")
    ] = None,
    want_slug: Annotated[
        bool,
        Field(True, description="Whether to include defillama_slug field")
    ] = True
) -> Dict[str, Any]:
    """
    Resolve DAO registry entry.
    Returns a record with governance token address (from Contracts), native token ticker (CMC_ticker),
    cmc_ucid, and DefiLlama name/slug.
    """
    args = LookupIn(
        space=space,
        name=name,
        ticker=ticker,
        contract=contract,
        want_slug=want_slug
    )

    rows = _load_rows()
    row = _match_row(rows, space=args.space, name=args.name, ticker=args.ticker, contract=args.contract)
    if not row:
        return {"found": False, "query": args.model_dump(), "note": "No matching row in registry"}

    contracts = [x.strip() for x in (row["contracts"].split("|") if row["contracts"] else []) if x.strip()]
    gov_addr = contracts[0] if contracts else None

    out: Dict[str, Any] = {
        "found": True,
        "space": row["space"] or None,
        "defillama_name": row["defillama_name"] or None,
        "defillama_slug": _slugify(row["defillama_name"]) if (args.want_slug and row["defillama_name"]) else None,
        "contracts": contracts or None,
        "cmc_ticker": row["cmc_ticker"] or None,
        "cmc_ucid": row["cmc_ucid"] or None,
        "cmc_link": row["cmc_link"] or None,
        "governance_token": {
            "address": gov_addr,
            "chain_id": 1 if gov_addr else None,
            "ticker": row["cmc_ticker"] or None,
            "cmc_ucid": row["cmc_ucid"] or None,
        },
        "native_token": {
            "ticker": row["cmc_ticker"] or None,
            "cmc_ucid": row["cmc_ucid"] or None,
        },
    }
    return out

@mcp.tool(
    name="health",
    title="Registry Service Health Check",
    description="Check the health status of the Registry MCP service. Returns service status and CSV data path information.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
def health() -> Dict[str, Any]:
    return {"ok": True, "csv": CSV_PATH}

if __name__ == "__main__":
    mcp.run(transport="stdio")
