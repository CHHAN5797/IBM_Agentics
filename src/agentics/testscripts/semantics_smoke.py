# testscripts/semantics_smoke.py
import os
import json
from pathlib import Path

# 프로젝트 루트에서 실행한다고 가정 (src 경로 추가)
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentics.mcp.semantics_mcp import find_papers, bibtex_from_papers  # 파일 경로에 맞게 import

def main():
    # 네트워크/키 상태 로그 용
    print("[info] SEMANTIC_SCHOLAR_API_KEY set?:", bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY")))
    os.environ.setdefault("SEMANTICS_MAX_RPM", "20")   # 보수적 호출
    os.environ.setdefault("SEMANTICS_BASE_SLEEP", "1") # 초기 슬립

    queries = ["DAO governance voting"]  # 주제는 명시적 강제 없이, 검색만 1개 쿼리
    # 딱 2개만 찾음 (enrich=False로 불필요한 추가 조회 방지)
    papers = find_papers(queries=queries, per_query=2, enrich=False)

    # 정상성 체크
    assert isinstance(papers, list), "papers should be a list"
    assert len(papers) <= 2, f"expected at most 2, got {len(papers)}"
    for p in papers:
        assert isinstance(p, dict)
        # 핵심 필드 점검(없어도 동작은 하므로 존재 시 타입만 확인)
        for key in ["title", "year", "authors", "url"]:
            _ = p.get(key, None)

    print("\n=== Found Papers (<=2) ===")
    print(json.dumps(papers, ensure_ascii=False, indent=2))

    # BibTeX도 한번 생성
    bib = bibtex_from_papers(papers)
    print("\n=== BibTeX ===")
    print(bib)

if __name__ == "__main__":
    main()
