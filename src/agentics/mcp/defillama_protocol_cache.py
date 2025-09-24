# -*- coding: utf-8 -*-
"""Helpers for DeFiLlama protocol metadata caching and slug resolution."""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

FetchFn = Callable[[], Iterable[Dict[str, Optional[str]]]]


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS protocols (
            slug TEXT PRIMARY KEY,
            name TEXT,
            symbol TEXT,
            last_refreshed_at INTEGER
        )
        """
    )
    conn.commit()
    return conn


def _normalize(text: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def protocol_count(db_path: Path) -> int:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM protocols;").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def slug_exists(db_path: Path, slug: str) -> bool:
    if not slug:
        return False
    with _connect(db_path) as conn:
        cur = conn.execute(
            "SELECT 1 FROM protocols WHERE slug=? LIMIT 1;", (slug.lower(),)
        )
        return cur.fetchone() is not None


def refresh_protocols_cache(
    db_path: Path, fetcher: FetchFn, ttl_hours: int = 24
) -> int:
    ttl_seconds = max(int(ttl_hours * 3600), 0)
    now = int(time.time())
    with _connect(db_path) as conn:
        row = conn.execute("SELECT MAX(last_refreshed_at) FROM protocols;").fetchone()
    last = int(row[0]) if row and row[0] is not None else 0
    if ttl_seconds and last and (now - last) < ttl_seconds:
        return 0

    records = list(fetcher())
    if not records:
        return 0

    with _connect(db_path) as conn:
        for rec in records:
            slug = (rec.get("slug") or "").strip().lower()
            if not slug:
                continue
            conn.execute(
                """
                INSERT INTO protocols(slug,name,symbol,last_refreshed_at)
                VALUES(?,?,?,?)
                ON CONFLICT(slug) DO UPDATE SET
                    name=excluded.name,
                    symbol=excluded.symbol,
                    last_refreshed_at=excluded.last_refreshed_at
                """,
                (slug, rec.get("name"), rec.get("symbol"), now),
            )
        conn.commit()
    return len(records)


def _build_hint_list(hints: Sequence[str]) -> List[Tuple[str, str]]:
    seen = set()
    ordered: List[Tuple[str, str]] = []
    for raw in hints:
        raw = (raw or "").strip().lower()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        ordered.append((raw, _normalize(raw)))
    return ordered


def rank_protocol_candidates(
    db_path: Path,
    hints: Sequence[str],
    limit: Optional[int] = None,
) -> List[Dict[str, Optional[str]]]:
    hint_pairs = _build_hint_list(hints)
    if not hint_pairs:
        return []

    results: List[Dict[str, Optional[str]]] = []
    with _connect(db_path) as conn:
        for slug, name, symbol in conn.execute(
            "SELECT slug,name,symbol FROM protocols;"
        ):
            score = 0
            matches: List[Dict[str, str]] = []
            for raw_hint, norm_hint in hint_pairs:
                if not norm_hint:
                    continue
                for field, value in ("slug", slug), ("name", name), ("symbol", symbol):
                    cand_norm = _normalize(value)
                    if not cand_norm:
                        continue
                    if norm_hint == cand_norm:
                        score += 4
                        matches.append(
                            {"field": field, "hint": raw_hint, "kind": "exact"}
                        )
                    elif norm_hint in cand_norm:
                        score += 2
                        matches.append(
                            {"field": field, "hint": raw_hint, "kind": "contains"}
                        )
                    elif cand_norm in norm_hint:
                        score += 1
                        matches.append(
                            {"field": field, "hint": raw_hint, "kind": "contained_by"}
                        )
            if score:
                # Deduplicate matches while preserving order.
                deduped: List[Dict[str, str]] = []
                seen_pair = set()
                for m in matches:
                    key = (m["field"], m["hint"], m["kind"])
                    if key in seen_pair:
                        continue
                    seen_pair.add(key)
                    deduped.append(m)
                results.append(
                    {
                        "slug": slug,
                        "name": name,
                        "symbol": symbol,
                        "score": score,
                        "matches": deduped,
                    }
                )

    results.sort(key=lambda r: (-int(r["score"] or 0), r["slug"] or ""))

    if limit is not None and limit >= 0:
        return results[:limit]
    return results
