# src/agentics/orchestra/dual_llm_runner.py
from __future__ import annotations
import json, os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from agentics.llm.clients import OpenAIClient, GrokClient, LLMResult

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_root_dir(root: Path) -> Path:
    """
    Ensure Decision_runs is a directory.
    - If a *file* with the same name exists, rename it to .bak and create the dir.
    """
    if root.exists():
        if root.is_file():
            backup = root.parent / f"{root.name}_{_now_tag()}.bak"
            root.rename(backup)
    # create as directory if missing
    root.mkdir(parents=True, exist_ok=True)
    return root

def _safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def _save_run(root: Path, engine_name: str, tag: str,
              prompt: List[Dict[str, str]],
              res: Optional[LLMResult],
              err: Optional[str]) -> Path:
    outdir = root / engine_name / tag
    outdir.mkdir(parents=True, exist_ok=True)

    if err is not None:
        _safe_write_text(outdir / "error.txt", err)
    if res is not None:
    
        raw_dict = res.raw if isinstance(res.raw, dict) else {}
        _safe_write_text(outdir / "response.json", json.dumps(raw_dict, indent=2, default=str))

        _safe_write_text(outdir / "response.md", res.content or "")

        meta = {
            "model": res.model,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "prompt_messages": prompt,
        }
        usage = raw_dict.get("usage")
        if usage:
            meta["usage"] = usage
        _safe_write_text(outdir / "meta.json", json.dumps(meta, indent=2, ensure_ascii=False))
    return outdir

def run_both_and_save(
    messages: List[Dict[str, str]],
    decision_root: str = "Decision_runs",
    openai_model: str = "gpt-4o-mini",
    grok_model: Optional[str] = "grok-2",
    enable_grok: Optional[bool] = None,
    **kwargs
) -> Dict[str, str]:
    """
    결과는 Decision_runs/openai/<timestamp>/..., Decision_runs/grokai/<timestamp>/... 에 저장.
    - 각 엔진이 실패해도 상대 폴더에 error.txt 남김.
    - decision_root가 '파일'로 존재하면 자동 백업 후 디렉터리 생성.
    """
    root = _ensure_root_dir(Path(decision_root))
    tag = _now_tag()

    paths: Dict[str, str] = {}

    # OpenAI (always-on)
    oai_res: Optional[LLMResult] = None
    oai_err: Optional[str] = None
    try:
        oai = OpenAIClient()
        oai_res = oai.chat(messages, model=openai_model, **kwargs)
    except Exception as e:
        oai_err = f"{type(e).__name__}: {e}"
    oai_dir = _save_run(root, "openai", tag, messages, oai_res, oai_err)
    paths["openai"] = str(oai_dir)

    grok_enabled = enable_grok
    if grok_enabled is None:
        grok_enabled = bool(grok_model)

    grok_dir: Optional[Path] = None
    if grok_enabled and grok_model:
        grok_res: Optional[LLMResult] = None
        grok_err: Optional[str] = None
        try:
            grok = GrokClient()
            grok_res = grok.chat(messages, model=grok_model, **kwargs)
        except Exception as e:
            grok_err = f"{type(e).__name__}: {e}"
        grok_dir = _save_run(root, "grokai", tag, messages, grok_res, grok_err)
        paths["grokai"] = str(grok_dir)
    else:
        paths["grokai"] = ""

    return paths

if __name__ == "__main__":
    msgs = [
        {"role": "system", "content": "You are a concise research assistant."},
        {"role": "user", "content": "Give me 3 bullet pros/cons of dual-LLM ensembles."}
    ]
    locs = run_both_and_save(
        messages=msgs,
        decision_root=os.getenv("DECISION_ROOT", "Decision_runs"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        grok_model=os.getenv("GROK_MODEL", "grok-2"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3"))
    )
    print("Saved:", json.dumps(locs, indent=2))
