#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smoke test (light): take up to N proposals (default 10) and fetch comments per proposal.
This avoids long server-side /fetch-comments calls and reduces timeouts.
"""

import os, json, argparse, time
from typing import List, Dict, Any
import requests
from tqdm import tqdm

try:
    import pandas as pd
except Exception:
    pd = None

DEFAULT_SPACES = [
    "aavedao.eth","arbitrumfoundation.eth","snapshot.dcl.eth","balancer.eth","cvx.eth",
    "1inch.eth","aurafinance.eth","lido-snapshot.eth","uniswapgovernance.eth","metislayer2.eth",
]

# ----------------- FS helpers -----------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def write_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)
def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
def write_csv(path: str, rows: List[Dict[str, Any]]):
    ensure_dir(os.path.dirname(path) or ".")
    if pd is not None:
        pd.DataFrame(rows).to_csv(path, index=False); return
    if not rows:
        open(path, "w").close(); return
    headers = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            vals = []
            for h in headers:
                v = r.get(h, ""); v = "" if v is None else str(v).replace('"','""')
                if ("," in v) or ("\n" in v): v = f"\"{v}\""
                vals.append(v)
            f.write(",".join(vals) + "\n")

# ----------------- HTTP with retries -----------------
def post_json(base_url: str, path: str, payload: Dict[str, Any], timeout: int = 60, retries: int = 5, backoff: float = 1.5):
    url = base_url.rstrip("/") + path
    last = None
    for i in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last = e; sleep_s = backoff**i
            print(f"[warn] POST {path} attempt {i+1}/{retries} failed ({e}); retrying in {sleep_s:.1f}s")
            time.sleep(sleep_s)
        except requests.RequestException as e:
            raise
    raise last

# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser(description="MCP smoke test (light): up to N proposals via /forums/fetch")
    ap.add_argument("--base-url", default="http://localhost:8000", help="MCP server base url")
    ap.add_argument("--spaces", default=",".join(DEFAULT_SPACES),
                    help="Comma-separated snapshot spaces")
    ap.add_argument("--limit-proposals", type=int, default=10, help="Max proposals to test (default: 10)")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds per request")
    ap.add_argument("--max-pages", type=int, default=5, help="Discourse pagination max pages")
    ap.add_argument("--max-posts", type=int, default=200, help="Max posts per proposal (discourse)")
    ap.add_argument("--use-cutoff", action="store_true",
                    help="If set, filter comments by proposal end timestamp")
    ap.add_argument("--outdir", default="MCP_test", help="Output directory")
    return ap.parse_args()

# ----------------- Main -----------------
def main():
    args = parse_args()
    base = args.base_url.rstrip("/")
    spaces = [s.strip() for s in args.spaces.split(",") if s.strip()]
    outdir = args.outdir
    ensure_dir(outdir); ensure_dir(os.path.join(outdir, "by_proposal"))

    print(f"▶ Base URL      : {base}")
    print(f"▶ Spaces        : {spaces}")
    print(f"▶ Limit         : {args.limit_proposals} proposals")
    print(f"▶ Timeout       : {args.timeout}s per request")
    print(f"▶ Max pages/post: {args.max_pages}/{args.max_posts}")
    print(f"▶ Use cutoff    : {args.use_cutoff}")
    print(f"▶ Outdir        : {outdir}\n")

    # 1) List proposals with discussion (batched = 1 for safety)
    print("1) /snapshot/with-discussion/list ...")
    list_all = []
    for s in spaces:
        try:
            part = post_json(base, "/snapshot/with-discussion/list", {"spaces":[s]}, timeout=args.timeout)
            list_all.extend(part)
            write_json(os.path.join(outdir, f"list__{s}.json"), part)
        except Exception as e:
            print(f"[error] list for {s} failed: {e}")

    # Save combined list
    write_json(os.path.join(outdir, "list.json"), list_all)
    write_csv(os.path.join(outdir, "list.csv"), list_all)
    print(f"   - proposals returned: {len(list_all)}\n")

    if not list_all:
        print("[exit] no proposals returned."); return

    # 2) Take up to N proposals (sample: first N)
    sample = list_all[: max(0, min(args.limit_proposals, len(list_all)))]
    write_json(os.path.join(outdir, "list_sample.json"), sample)
    write_csv(os.path.join(outdir, "list_sample.csv"), sample)
    print(f"2) Took {len(sample)} proposals for fetching.\n")

    # 3) Fetch comments per proposal using /forums/fetch
    print("3) /forums/fetch per proposal ...")
    all_rows = []       # flattened rows for CSV
    all_jsonl = []      # for LLM
    for it in tqdm(sample, desc="fetch"):
        url = (it.get("discussion_url") or "").strip()
        if not url:
            continue
        payload = {
            "url": url,
            "max_pages": args.max_pages,
            "max_posts": args.max_posts,
        }
        if args.use_cutoff:
            payload["end_ts_filter"] = int(it.get("end") or 0)

        try:
            data = post_json(base, "/forums/fetch", payload, timeout=args.timeout)
        except Exception as e:
            print(f"[warn] fetch failed for {url}: {e}")
            continue

        # Discourse → posts를 평탄화
        if data.get("type") == "discourse":
            posts = data.get("posts") or []
            # per-proposal CSV
            per_rows = []
            for p in posts:
                text = p.get("raw") or p.get("cooked") or ""
                row = {
                    "space": it.get("space"),
                    "proposal_id": it.get("id"),
                    "proposal_title": it.get("title"),
                    "discussion_url": url,
                    "end": it.get("end"),
                    "end_iso": it.get("end_iso"),
                    "forum_kind": "discourse",
                    "post_id": p.get("id"),
                    "post_number": p.get("post_number"),
                    "reply_to_post_number": p.get("reply_to_post_number"),
                    "author_username": p.get("username"),
                    "created_at": p.get("created_at"),
                    "text": text,
                    "text_raw": p.get("raw"),
                    "text_html": p.get("cooked"),
                }
                per_rows.append(row)
                all_rows.append(row)
                all_jsonl.append({"text": text, "meta": {k:v for k,v in row.items() if k!="text"}})

            if per_rows:
                per_path = os.path.join(outdir, "by_proposal", f"{it.get('space')}__{it.get('id')}__comments.csv")
                write_csv(per_path, per_rows)

        else:
            # generic 페이지는 스킵 (원한다면 상태만 기록)
            pass

    # 4) save combined outputs
    write_csv(os.path.join(outdir, "comments_all.csv"), all_rows)
    write_jsonl(os.path.join(outdir, "comments_all.jsonl"), all_jsonl)

    # summary
    total_posts = len(all_rows)
    summary = [
        f"Base URL: {base}",
        f"Spaces: {', '.join(spaces)}",
        f"Limit proposals: {len(sample)}",
        f"Timeout per req: {args.timeout}s",
        f"Max pages/posts: {args.max_pages}/{args.max_posts}",
        f"Use cutoff: {args.use_cutoff}",
        "",
        f"Total discourse comments: {total_posts}",
        "Outputs:",
        " - list.json / list.csv (all proposals with discussion)",
        " - list_sample.json / list_sample.csv (chosen subset)",
        " - comments_all.csv / comments_all.jsonl",
        " - by_proposal/<space>__<proposal_id>__comments.csv",
    ]
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print("✅ Done! See MCP_test/ for artifacts.")

if __name__ == "__main__":
    main()
