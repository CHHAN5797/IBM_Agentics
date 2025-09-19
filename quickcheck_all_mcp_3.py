# quickcheck_all_mcp.py
# End-to-end smoke test for three FastMCP servers:
# 1) SnapshotAPI  2) ForumsMCP  3) TimelineMCP
# - Input: SNAPSHOT_URL (env or prompt)
# - Flow:
#   a) Spawn SnapshotAPI -> resolve proposal id -> meta/result -> all votes (flattened) -> basic stats
#   b) If discussion URL exists -> spawn ForumsMCP -> fetch discussion/page
#   c) Spawn TimelineMCP -> analyze timeline metrics (no final tally)
#
# This script is version-tolerant for mcp client differences (stdio_connect vs stdio_client)
# and for multi-part tool results (unwrap_all + coerce_votes).

import os
import json
import asyncio
from typing import Any, AsyncIterator, List, Dict

from mcp import StdioServerParameters
from mcp.client.session import ClientSession

# Polyfill: stdio_connect (newer) vs stdio_client (older)
try:
    from mcp.client.stdio import stdio_connect as _stdio_connect_like
    _CONNECT_NAME = "stdio_connect"
except ImportError:
    from mcp.client.stdio import stdio_client as _stdio_connect_like  # type: ignore
    _CONNECT_NAME = "stdio_client"


# ---------- Utilities ----------

def pretty(x: Any) -> str:
    """Safe pretty printer for JSON-like structures."""
    try:
        return json.dumps(x, ensure_ascii=False, indent=2)
    except Exception:
        return repr(x)

def unwrap_one(res: Any) -> Any:
    """
    Unwrap a single CallToolResult into plain JSON if possible.
    (Keeps only the FIRST content part.)
    """
    content = getattr(res, "content", None)
    if isinstance(content, list) and content:
        part = content[0]
        data = getattr(part, "data", None)
        if data is not None:
            return data
        text = getattr(part, "text", None)
        if isinstance(text, str):
            try:
                return json.loads(text)
            except Exception:
                return text
    return res

def unwrap_all(res: Any) -> Any:
    """
    Gather ALL content parts from a CallToolResult into a single Python value.

    Rules:
    - If all parts are JSON lists, flatten into one list.
    - If all parts are JSON dicts, return a list[dict] (e.g., streamed records).
    - If parts are strings, try json.loads; keep as str otherwise.
    - If only ONE part exists, return that parsed part.
    - Fall back to the raw result if no .content exists.
    """
    content = getattr(res, "content", None)
    if not (isinstance(content, list) and content):
        return res  # already plain

    parsed = []
    for part in content:
        data = getattr(part, "data", None)
        if data is not None:
            parsed.append(data)
            continue
        text = getattr(part, "text", None)
        if isinstance(text, str):
            try:
                parsed.append(json.loads(text))
            except Exception:
                parsed.append(text)

    if len(parsed) == 1:
        return parsed[0]

    if all(isinstance(x, list) for x in parsed):
        out = []
        for x in parsed:
            out.extend(x)
        return out

    if all(isinstance(x, dict) for x in parsed):
        return parsed

    return parsed  # mixed types; return as-is

def coerce_votes(v: Any) -> List[Dict]:
    """
    Force votes into list[dict] for downstream safety.
    - dict -> [dict]
    - str  -> try json.loads
    - list -> ensure every element is dict (if string elements, try json.loads)
    - else -> []
    """
    if isinstance(v, dict):
        return [v]
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return []
    if isinstance(v, list):
        out: List[Dict] = []
        for item in v:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, str):
                try:
                    decoded = json.loads(item)
                    if isinstance(decoded, dict):
                        out.append(decoded)
                    elif isinstance(decoded, list):
                        out.extend([e for e in decoded if isinstance(e, dict)])
                except Exception:
                    pass
        return out
    return []


async def open_session(script_path: str) -> AsyncIterator[ClientSession]:
    """
    Open an MCP stdio session for a given server script.
    This is tolerant to (read, write) vs direct session returns from the connector.
    """
    params = StdioServerParameters(
        command="python3",
        args=[script_path],
        env={"PYTHONPATH": "src", **os.environ},
    )
    async with _stdio_connect_like(params) as conn:
        # Case 1: (read_stream, write_stream)
        if isinstance(conn, tuple) and len(conn) == 2:
            read, write = conn
            async with ClientSession(read, write) as session:
                if hasattr(session, "initialize"):
                    await session.initialize()
                yield session
            return

        # Case 2: already returns a session-like
        session = conn  # type: ignore
        if hasattr(session, "initialize"):
            await session.initialize()
        if hasattr(session, "__aenter__"):
            async with session:
                yield session
        else:
            yield session


# ---------- Main flow ----------

async def main():
    # 0) Input URL
    snapshot_url = os.getenv("SNAPSHOT_URL", "").strip()
    if not snapshot_url:
        snapshot_url = input("Snapshot URL (https://snapshot.org/#/<space>/proposal/<id>): ").strip()
    if not snapshot_url:
        raise SystemExit("A Snapshot URL is required.")

    # 1) SnapshotAPI
    async for snap in open_session("src/agentics/mcp/snapshot_api.py"):
        tools = [t.name for t in (await snap.list_tools()).tools]
        print(f"[SnapshotAPI] tools: {tools}")

        health = unwrap_one(await snap.call_tool("health", {}))
        print("[SnapshotAPI] health:", pretty(health))

        rid = unwrap_one(await snap.call_tool("resolve_proposal_id_from_url", {"snapshot_url": snapshot_url}))
        if not isinstance(rid, str) or not rid:
            raise SystemExit(f"Could not parse proposal id from URL.\nGot: {rid!r}")
        pid = rid
        print("[SnapshotAPI] proposal id:", pid)

        meta   = unwrap_one(await snap.call_tool("get_proposal_by_id", {"proposal_id": pid}))
        result = unwrap_one(await snap.call_tool("get_proposal_result_by_id", {"proposal_id": pid}))

        # Get ALL votes and normalize to list[dict]
        votes_res = await snap.call_tool("get_votes_all", {"proposal_id": pid, "batch": 500})
        votes_raw = unwrap_all(votes_res)
        votes = coerce_votes(votes_raw)

        print("\n=== META ===")
        print(pretty(meta))
        print("\n=== RESULT (reporting only) ===")
        print(pretty(result))
        print("\n=== ALL VOTES ===")
        print("count:", len(votes))
        if votes[:3]:
            print("sample:", pretty(votes[:3]))

        # Basic stats
        voters = [(v.get("voter") or "").lower() for v in votes if v.get("voter")]
        unique_voters = {a for a in voters if a}
        print("unique voters:", len(unique_voters))

        # Keep for later stages
        start = int(meta.get("start") or 0) if isinstance(meta, dict) else 0
        end = int(meta.get("end") or 0) if isinstance(meta, dict) else 0
        choices = meta.get("choices") if isinstance(meta, dict) else []
        discussion_url = (meta.get("discussion") or "").strip() if isinstance(meta, dict) else ""

    # 2) ForumsMCP (if discussion_url exists)
    if discussion_url:
        try:
            async for forums in open_session("src/agentics/mcp/forums_mcp.py"):
                tools = [t.name for t in (await forums.list_tools()).tools]
                print(f"\n[ForumsMCP] tools: {tools}")
                h = unwrap_one(await forums.call_tool("health", {}))
                print("[ForumsMCP] health:", pretty(h))

                if "/t/" in discussion_url:
                    disc = unwrap_all(await forums.call_tool("fetch_discussion", {
                        "url": discussion_url, "max_pages": 3
                    }))
                else:
                    disc = unwrap_one(await forums.call_tool("fetch_page", {
                        "url": discussion_url, "max_bytes": 200_000
                    }))

                print("\n=== DISCUSSION ===")
                print(pretty(disc))
        except FileNotFoundError:
            print("\n[ForumsMCP] skipped (server file not found).")
        except Exception as e:
            print(f"\n[ForumsMCP] skipped due to error: {e!r}")
    else:
        print("\n[ForumsMCP] skipped (no discussion URL on proposal).")

    # 3) TimelineMCP
    try:
        async for tl in open_session("src/agentics/mcp/timeline_mcp.py"):
            tools = [t.name for t in (await tl.list_tools()).tools]
            print(f"\n[TimelineMCP] tools: {tools}")
            h = unwrap_one(await tl.call_tool("health", {}))
            print("[TimelineMCP] health:", pretty(h))

            analysis = unwrap_one(await tl.call_tool("analyze_timeline", {
                "start": start, "end": end, "choices": choices, "votes": votes
            }))
            print("\n=== TIMELINE ANALYSIS ===")
            print(pretty(analysis))
            rec = analysis.get("recommended_index") if isinstance(analysis, dict) else None
            if isinstance(rec, int) and isinstance(choices, list) and 0 <= rec < len(choices):
                print("recommended choice:", f"{rec} → {choices[rec]}")
            print("\nOK ✅  (all three MCP servers checked)")
    except FileNotFoundError:
        print("\n[TimelineMCP] skipped (server file not found).")
    except Exception as e:
        print(f"\n[TimelineMCP] error: {e!r}")

if __name__ == "__main__":
    # To avoid noisy teardown issues, catch top-level exceptions gracefully.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nFatal error: {e!r}")
