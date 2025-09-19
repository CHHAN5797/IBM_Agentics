#!/usr/bin/env python3
"""
MCP (Snapshot ↔ Governance Forums) 통합 테스트 스크립트

요구사항:
- MCP 서버 실행 중 (기본: http://localhost:8000)
- 서버는 /snapshot/finished-proposals 가 proposal body/body_text를 포함하도록 패치되어 있어야 함
- 서버는 /forums/fetch 가 discourse 게시물에 대해 post["text"] (HTML 제거) 를 포함하도록 패치되어 있으면 더 좋음

사용:
    python test.py
환경변수:
    MCP_BASE_URL = "http://localhost:8000" (기본)
"""

import os
import sys
import time
import json
import re
from collections import defaultdict

import requests

# -----------------------------
# Config
# -----------------------------
BASE = os.environ.get("MCP_BASE_URL", "http://localhost:8000")

# 테스트할 Snapshot spaces (필요하면 수정/추가)
SPACES = [
    "aavedao.eth",
    "uniswapgovernance.eth",
    # "arbitrumfoundation.eth",
    # "balancer.eth",
    # "1inch.eth",
    # "lido-snapshot.eth",
]

# 각 space에서 상위 몇 개의 finished proposals만 테스트할지 (클라이언트 측 제한)
TOP_K_PER_SPACE = 5

# HTTP 타임아웃 (초)
TIMEOUT_PROPOSALS = 240
TIMEOUT_FORUM = 240

# 출력 파일
OUT_PROPOSALS_JSON = "sample_finished_proposals.json"
OUT_FORUM_JSON = "sample_forum_thread.json"


# -----------------------------
# Helpers
# -----------------------------
HTML_TAG_RE = re.compile(r"<[^>]+>")

def strip_html(s: str) -> str:
    if not s:
        return ""
    return HTML_TAG_RE.sub("", s).strip()

def pretty(obj, width=2000):
    print(json.dumps(obj, indent=2, ensure_ascii=False)[:width], "\n...")


# -----------------------------
# Calls
# -----------------------------
def health_check():
    url = f"{BASE}/health"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def fetch_finished_proposals(spaces):
    url = f"{BASE}/snapshot/finished-proposals"
    r = requests.post(url, json={"spaces": spaces}, timeout=TIMEOUT_PROPOSALS)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response: {data}")
    return data

def fetch_forum(url, end_ts):
    api = f"{BASE}/forums/fetch"
    r = requests.post(api, json={"url": url, "end_ts_filter": end_ts}, timeout=TIMEOUT_FORUM)
    r.raise_for_status()
    return r.json()


# -----------------------------
# Main
# -----------------------------
def main():
    print(f"[1] Health check → {BASE}/health")
    try:
        h = health_check()
        print("health:", h)
    except Exception as e:
        print("Health check failed:", e)
        sys.exit(1)

    print("\n[2] Fetching finished proposals ...")
    proposals = fetch_finished_proposals(SPACES)
    print(f"finished proposals returned: {len(proposals)}")

    # 클라이언트 측에서 space별 상위 N개 제한
    per_space = defaultdict(list)
    for p in proposals:
        per_space[p.get("space")].append(p)

    # Snapshot 쿼리는 created desc라 가정 → slice
    limited = []
    for s in SPACES:
        subset = per_space.get(s, [])[:TOP_K_PER_SPACE]
        limited.extend(subset)

    print(f"limited to top {TOP_K_PER_SPACE} per space → total {len(limited)}")
    print("\n== Sample proposals ==")
    pretty(limited[:3])

    # 간단 검증: body/body_text 존재 확인
    missing_body = [p for p in limited if not p.get("body")]
    missing_body_text = [p for p in limited if not p.get("body_text")]
    if missing_body:
        print(f"[WARN] {len(missing_body)} proposals have no 'body' (server patch needed?). Showing first:")
        pretty(missing_body[:1])
    if missing_body_text:
        print(f"[WARN] {len(missing_body_text)} proposals have no 'body_text' (server patch needed?). Showing first:")
        pretty(missing_body_text[:1])

    # 저장
    with open(OUT_PROPOSALS_JSON, "w", encoding="utf-8") as f:
        json.dump(limited, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {OUT_PROPOSALS_JSON} ({len(limited)} items)")

    # discussion_url 있는 것 중 하나 테스트
    with_forum = [p for p in limited if p.get("discussion_url")]
    if not with_forum:
        print("\n[INFO] No proposals with discussion_url in limited set; expand SPACES or increase TOP_K_PER_SPACE.")
        return

    sample = with_forum[0]
    print("\n[3] Testing forums/fetch on discussion_url:")
    print("space:", sample["space"])
    print("title:", sample["title"])
    print("author:", sample.get("author"))
    print("end_iso:", sample.get("end_iso"))
    print("discussion_url:", sample["discussion_url"])

    forum = fetch_forum(sample["discussion_url"], sample["end"])
    print("\n== forum fetch result keys:", list(forum.keys()))
    print("type:", forum.get("type"))

    # Discourse라면 posts_returned/텍스트 확인
    if forum.get("type") == "discourse":
        th = forum.get("thread", {}) or {}
        posts = forum.get("posts", []) or []
        print("thread.title:", th.get("title"))
        print("posts_returned:", forum.get("posts_returned"))
        # text 필드가 없을 수도 있으니 strip_html로 대체 지원
        first = posts[0] if posts else {}
        text = first.get("text") or strip_html(first.get("cooked") or "") or first.get("raw")
        print("sample post:", {
            "id": first.get("id"),
            "username": first.get("username"),
            "created_at": first.get("created_at"),
            "text": (text[:200] + "…") if text else None,
        })
    else:
        # generic HTML 페이지의 경우
        print("generic page status:", forum.get("status"))
        content_len = len(forum.get("content", "") or "")
        print("generic page content length:", content_len)

    # 저장
    with open(OUT_FORUM_JSON, "w", encoding="utf-8") as f:
        json.dump(forum, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {OUT_FORUM_JSON}")

    # 추가 체크: LLM 비사용 확인(서버는 외부 LLM 호출 없음)
    print("\n[4] Sanity check: This test used only HTTP calls to MCP; no LLM calls were made by this script.")
    print("Done.")

if __name__ == "__main__":
    main()
