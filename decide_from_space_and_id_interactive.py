#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive runner (with back navigation + 'all pages' option + semantics toggle)
- Calls MCP: /snapshot/analyze-from-url
- Saves:
    <proposal_id>__response.json
    <proposal_id>__compare.md
    <proposal_id>__prompt_ascii.txt
    <proposal_id>__llm_response.txt
    (semantics prompt/response are inside response.json + printed in compare.md)
"""

import os
import re
import json
import requests
import unicodedata
from pathlib import Path
from typing import Optional

# -------- .env loader (optional) --------
try:
    from dotenv import load_dotenv
    dotenv_path = Path(".env")
    dotenv_local = Path(".env.local")
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    if dotenv_local.exists():
        load_dotenv(dotenv_path=dotenv_local, override=True)
except Exception:
    pass

# -------- Defaults --------
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_PAGES = 6
DEFAULT_KEEP = 20
DEFAULT_BODY_MAX = 3000
DEFAULT_OUTDIR = "Decision_runs"

# --- NEW: semantics defaults (change via wizard) ---
DEFAULT_CALL_SEMANTICS = False
DEFAULT_SEM_PROMPT_MODE = "build"   # or "call"
DEFAULT_SEM_LIMIT = 12
DEFAULT_SEM_YEAR_FROM = 2021
DEFAULT_SEM_YEAR_TO = 2025
DEFAULT_SEM_FOCAL_DOI = "10.2139/ssrn.4367209"
DEFAULT_SEM_TOPIC = "Decentralized Governance; DAO; on-chain voting; delegation; quadratic voting"

SNAPSHOT_URL_RE = re.compile(r"https?://[^ ]*/#/.+?/proposal/([0-9a-zA-Z]+)")
_ZW_CTRL_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]+|[\x00-\x08\x0B\x0C\x0E-\x1F]")

# -------- Helpers --------
def _clean_utf8(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return "" if s is None else str(s)
    s = unicodedata.normalize("NFC", s)
    s = _ZW_CTRL_RE.sub("", s)
    return s

def parse_proposal_id_from_url(url: str) -> str | None:
    m = SNAPSHOT_URL_RE.search((url or "").strip())
    return m.group(1) if m else None

def build_snapshot_url(space: str, proposal_id: str) -> str:
    space = (space or "").strip().rstrip("/")
    pid = (proposal_id or "").strip()
    return f"https://snapshot.org/#/{space}/proposal/{pid}"

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def call_mcp_analyze(base_url: str, payload: dict, timeout: int = 300) -> dict:
    url = base_url.rstrip("/") + "/snapshot/analyze-from-url"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ---- Input helpers with back/quit ----
def prompt_text(label: str, default: Optional[str], show_current: bool, current_val: Optional[str]) -> str:
    parts = []
    if show_current and current_val not in (None, ""):
        parts.append(f"current='{current_val}'")
    if default not in (None, ""):
        parts.append(f"default='{default}'")
    hint = (" [" + ", ".join(parts) + "]") if parts else ""
    return input(f"{label}{hint} (b=back, q=quit): ").strip()

def prompt_yesno(label: str, default: Optional[bool], current: Optional[bool]) -> str:
    def2txt = {True: "Y", False: "N", None: ""}
    cur2txt = {True: "Y", False: "N", None: ""}
    parts = []
    if current is not None:
        parts.append(f"current={cur2txt[current]}")
    if default is not None:
        parts.append(f"default={def2txt[default]}")
    hint = (" [" + ", ".join(parts) + "]") if parts else ""
    return input(f"{label}{hint} (y/n, b=back, q=quit): ").strip().lower()

def is_back(s: str) -> bool:
    return s.lower() == "b"

def is_quit(s: str) -> bool:
    return s.lower() == "q"

def coalesce(val: Optional[str], current: Optional[str], default: Optional[str]) -> str:
    if val is None or val == "":
        if current not in (None, ""):
            return current
        return default or ""
    return val

def coalesce_bool(val: Optional[str], current: Optional[bool], default: Optional[bool]) -> Optional[bool]:
    if val in ("y", "yes"):
        return True
    if val in ("n", "no"):
        return False
    if val in ("", None):
        return current if current is not None else default
    return None

# ---- Wizard steps ----
def wizard():
    answers = {
        "base_url": DEFAULT_BASE_URL,
        "use_full_url": True,
        "snapshot_url": "",
        "space": "",
        "proposal_id": "",
        "pages": str(DEFAULT_PAGES),
        "keep": str(DEFAULT_KEEP),
        "body_max": str(DEFAULT_BODY_MAX),
        "call_llm": False,
        "outdir": DEFAULT_OUTDIR,

        # NEW: semantics toggles/params
        "call_semantics": DEFAULT_CALL_SEMANTICS,
        "sem_prompt_mode": DEFAULT_SEM_PROMPT_MODE,
        "sem_limit": str(DEFAULT_SEM_LIMIT),
        "sem_year_from": str(DEFAULT_SEM_YEAR_FROM),
        "sem_year_to": str(DEFAULT_SEM_YEAR_TO),
        "sem_focal_doi": DEFAULT_SEM_FOCAL_DOI,
        "sem_topic": DEFAULT_SEM_TOPIC,
    }

    steps = [
        "base_url",
        "use_full_url",
        "pages",
        "keep",
        "body_max",
        "call_llm",
        "call_semantics",
        "sem_params",
        "outdir",
        "confirm",
    ]

    idx = 0
    while 0 <= idx < len(steps):
        step = steps[idx]

        if step == "base_url":
            raw = prompt_text("MCP server URL", DEFAULT_BASE_URL, True, answers["base_url"])
            if is_quit(raw): return None
            if is_back(raw): continue
            answers["base_url"] = coalesce(raw, answers["base_url"], DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
            idx += 1
            continue

        if step == "use_full_url":
            raw = prompt_yesno("Do you want to paste a full Snapshot URL?", True, answers["use_full_url"])
            if is_quit(raw): return None
            if is_back(raw):
                idx -= 1
                continue
            yn = coalesce_bool(raw, answers["use_full_url"], True)
            if yn is None:
                print("  → Please answer 'y' or 'n', or use 'b'/'q'.")
                continue
            answers["use_full_url"] = yn

            if answers["use_full_url"]:
                while True:
                    raw2 = prompt_text("Snapshot URL (e.g., https://snapshot.org/#/<space>/proposal/<id>)", None, True, answers["snapshot_url"])
                    if is_quit(raw2): return None
                    if is_back(raw2): break
                    cand = coalesce(raw2, answers["snapshot_url"], None).strip()
                    if parse_proposal_id_from_url(cand):
                        answers["snapshot_url"] = cand
                        idx += 1
                        break
                    else:
                        print("  → Invalid URL; could not detect proposal id. Try again (or 'b' to go back).")
                if is_back(raw2):
                    continue
            else:
                while True:
                    raw2 = prompt_text("Snapshot space (e.g., aavedao.eth)", None, True, answers["space"])
                    if is_quit(raw2): return None
                    if is_back(raw2): break
                    space = coalesce(raw2, answers["space"], None).strip()
                    if space and "." in space:
                        answers["space"] = space
                        while True:
                            raw3 = prompt_text("Proposal id (0x... or alphanumeric hash)", None, True, answers["proposal_id"])
                            if is_quit(raw3): return None
                            if is_back(raw3): break
                            pid = coalesce(raw3, answers["proposal_id"], None).strip()
                            if pid and re.match(r"^[0-9a-zA-Z]+$", pid):
                                answers["proposal_id"] = pid
                                answers["snapshot_url"] = build_snapshot_url(space, pid)
                                idx += 1
                                break
                            else:
                                print("  → Invalid proposal id. Only hex/alphanumeric is allowed.")
                        if is_back(raw3):
                            continue
                        break
                    else:
                        print("  → Invalid space format. Example: aavedao.eth")
                if is_back(raw2):
                    continue
            continue

        if step == "pages":
            label = "Max Discourse pages to fetch (enter 'all' for no limit, b=back, q=quit)"
            raw = prompt_text(label, str(DEFAULT_PAGES), False, None)
            if is_quit(raw): return None
            if is_back(raw):
                idx -= 1
                continue
            raw = raw.strip().lower()
            if raw == "":
                answers["pages"] = str(DEFAULT_PAGES)
                idx += 1
                continue
            if raw == "all":
                answers["pages"] = str(-1)
                idx += 1
                continue
            try:
                answers["pages"] = str(int(raw))
                idx += 1
            except ValueError:
                print("  → Enter an integer or 'all'.")
            continue

        if step == "keep":
            raw = prompt_text("Max number of comments to pass to LLM", str(DEFAULT_KEEP), True, answers["keep"])
            if is_quit(raw): return None
            if is_back(raw):
                idx -= 1
                continue
            try:
                answers["keep"] = str(int(coalesce(raw, answers["keep"], str(DEFAULT_KEEP))))
                idx += 1
            except ValueError:
                print("  → Please enter an integer.")
            continue

        if step == "body_max":
            raw = prompt_text("Max characters for proposal body", str(DEFAULT_BODY_MAX), True, answers["body_max"])
            if is_quit(raw): return None
            if is_back(raw):
                idx -= 1
                continue
            try:
                answers["body_max"] = str(int(coalesce(raw, answers["body_max"], str(DEFAULT_BODY_MAX))))
                idx += 1
            except ValueError:
                print("  → Please enter an integer.")
            continue

        if step == "call_llm":
            env_key = os.getenv("OPENAI_API_KEY") or ""
            env_model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
            cur = answers["call_llm"]
            label = f"Call the LLM (OpenAI) as well?  [Model detected: {env_model}]"
            raw = prompt_yesno(label, False, cur)
            if is_quit(raw): return None
            if is_back(raw):
                idx -= 1
                continue
            yn = coalesce_bool(raw, cur, False)
            if yn is None:
                print("  → Please answer 'y' or 'n', or use 'b'/'q'.")
                continue
            if yn and not env_key:
                print("  ⚠️  OPENAI_API_KEY not found. Proceeding WITHOUT LLM call.")
                yn = False
            answers["call_llm"] = yn
            idx += 1
            continue

        if step == "call_semantics":
            raw = prompt_yesno("Call Semantics (Semantic Scholar MCP)?", DEFAULT_CALL_SEMANTICS, answers["call_semantics"])
            if is_quit(raw): return None
            if is_back(raw):
                idx -= 1
                continue
            yn = coalesce_bool(raw, answers["call_semantics"], DEFAULT_CALL_SEMANTICS)
            if yn is None:
                print("  → Please answer 'y' or 'n'.")
                continue
            answers["call_semantics"] = yn
            idx += 1
            continue

        if step == "sem_params":
            if not answers["call_semantics"]:
                idx += 1
                continue
            # prompt_mode
            raw = prompt_text("Semantics prompt mode ('build' or 'call')", DEFAULT_SEM_PROMPT_MODE, True, answers["sem_prompt_mode"])
            if is_quit(raw): return None
            if is_back(raw):
                idx -= 1
                continue
            pm = (raw or answers["sem_prompt_mode"] or DEFAULT_SEM_PROMPT_MODE).strip().lower()
            if pm not in ("build", "call"):
                pm = DEFAULT_SEM_PROMPT_MODE
            answers["sem_prompt_mode"] = pm

            # limit
            raw = prompt_text("Semantics limit_recent", str(DEFAULT_SEM_LIMIT), True, answers["sem_limit"])
            if is_quit(raw): return None
            if is_back(raw):
                idx -= 1
                continue
            try:
                answers["sem_limit"] = str(int(coalesce(raw, answers["sem_limit"], str(DEFAULT_SEM_LIMIT))))
            except ValueError:
                answers["sem_limit"] = str(DEFAULT_SEM_LIMIT)

            # years
            raw = prompt_text("Semantics year_from", str(DEFAULT_SEM_YEAR_FROM), True, answers["sem_year_from"])
            if is_quit(raw): return None
            if is_back(raw):
                idx -= 1
                continue
            try:
                answers["sem_year_from"] = str(int(coalesce(raw, answers["sem_year_from"], str(DEFAULT_SEM_YEAR_FROM))))
            except ValueError:
                answers["sem_year_from"] = str(DEFAULT_SEM_YEAR_FROM)

            raw = prompt_text("Semantics year_to", str(DEFAULT_SEM_YEAR_TO), True, answers["sem_year_to"])
            if is_quit(raw): return None
            if is_back(raw):
                idx -= 1
                continue
            try:
                answers["sem_year_to"] = str(int(coalesce(raw, answers["sem_year_to"], str(DEFAULT_SEM_YEAR_TO))))
            except ValueError:
                answers["sem_year_to"] = str(DEFAULT_SEM_YEAR_TO)

            # focal doi
            raw = prompt_text("Semantics focal DOI", DEFAULT_SEM_FOCAL_DOI, True, answers["sem_focal_doi"])
            if is_quit(raw): return None
            if is_back(raw):
                idx -= 1
                continue
            answers["sem_focal_doi"] = coalesce(raw, answers["sem_focal_doi"], DEFAULT_SEM_FOCAL_DOI)

            # topic
            raw = prompt_text("Semantics topic", DEFAULT_SEM_TOPIC, True, answers["sem_topic"])
            if is_quit(raw): return None
            if is_back(raw):
                idx -= 1
                continue
            answers["sem_topic"] = coalesce(raw, answers["sem_topic"], DEFAULT_SEM_TOPIC)
            idx += 1
            continue

        if step == "outdir":
            raw = prompt_text("Output directory", DEFAULT_OUTDIR, True, answers["outdir"])
            if is_quit(raw): return None
            if is_back(raw):
                idx -= 1
                continue
            answers["outdir"] = coalesce(raw, answers["outdir"], DEFAULT_OUTDIR).strip() or DEFAULT_OUTDIR
            idx += 1
            continue

        if step == "confirm":
            print("\n▶ Run summary")
            print(f"- MCP server     : {answers['base_url']}")
            print(f"- Snapshot URL   : {answers['snapshot_url']}")
            pages_disp = "all" if int(answers['pages']) <= 0 else answers['pages']
            print(f"- pages/keep     : {pages_disp} / {answers['keep']}")
            print(f"- body_max       : {answers['body_max']}")
            print(f"- call LLM       : {answers['call_llm']}")
            print(f"- call Semantics : {answers['call_semantics']}")
            if answers["call_semantics"]:
                print(f"  · sem prompt_mode={answers['sem_prompt_mode']}, limit={answers['sem_limit']}, years={answers['sem_year_from']}–{answers['sem_year_to']}")
                print(f"  · sem focal_doi={answers['sem_focal_doi']}")
                print(f"  · sem topic    = {answers['sem_topic']}")
            print(f"- outdir         : {answers['outdir']}\n")
            raw = prompt_yesno("Proceed?", True, None)
            if is_quit(raw): return None
            if is_back(raw):
                idx -= 1
                continue
            yn = coalesce_bool(raw, None, True)
            if yn:
                return answers
            else:
                print("Aborted by user.")
                return None

    return None

# -------- Main --------
def main():
    print("\n=== Snapshot Governance Interactive Runner (with 'all pages' + back + semantics) ===\n")
    answers = wizard()
    if not answers:
        print("Exited.")
        return

    pages = int(answers["pages"])
    keep = int(answers["keep"])
    body_max = int(answers["body_max"])

    outdir = answers["outdir"]
    Path(outdir).mkdir(parents=True, exist_ok=True)

    try:
        payload = {
            "snapshot_url": answers["snapshot_url"],
            "max_comment_pages": int(pages),  # <=0 means no limit on server
            "max_comment_keep": keep,
            "body_max_chars": body_max,
            "call_llm": bool(answers["call_llm"]),

            # NEW: semantics
            "call_semantics": bool(answers["call_semantics"]),
            "semantics_prompt_mode": answers["sem_prompt_mode"],
            "semantics_limit_recent": int(answers["sem_limit"]),
            "semantics_year_from": int(answers["sem_year_from"]),
            "semantics_year_to": int(answers["sem_year_to"]),
            "semantics_focal_doi": answers["sem_focal_doi"],
            "semantics_topic": answers["sem_topic"],
        }
        resp = call_mcp_analyze(
            base_url=answers["base_url"],
            payload=payload,
            timeout=480
        )
    except requests.Timeout:
        print("❌ Request timed out. Try reducing pages/keep or check the server.")
        return
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return

    pid = resp.get("proposal_id") or "unknown"
    run_base = Path(outdir) / pid

    with open(str(run_base) + "__response.json", "w", encoding="utf-8") as f:
        json.dump(resp, f, ensure_ascii=False, indent=2)

    comp = resp.get("comparison") or {}
    actual = comp.get("actual") or {}
    simulated = comp.get("simulated") or {}
    llm = comp.get("llm") or {}
    matches = comp.get("matches") or {}
    vsum = (resp.get("vote_summary") or {}) if isinstance(resp.get("vote_summary"), dict) else {}

    lines = []
    lines.append(f"# Proposal Comparison\n")
    lines.append(f"**Proposal ID:** {pid}")
    lines.append(f"**Title:** { (resp.get('proposal_title') or '').strip() }")
    lines.append("")
    lines.append("## Winners / Decisions")
    lines.append(f"- **Actual winner**: index={actual.get('winner_index')} label={actual.get('winner_label')} state={actual.get('state')}")
    lines.append(f"- **Simulated winner**: index={simulated.get('winner_index')} label={simulated.get('winner_label')}")
    lines.append(f"- **LLM decision**: index={llm.get('choice_index')} label={llm.get('choice_label')}")
    lines.append("")
    lines.append("## Matches")
    lines.append(f"- Simulated vs Actual: **{ '✅' if matches.get('simulated_vs_actual') else '❌' }**")
    lines.append(f"- LLM vs Actual: **{ '✅' if matches.get('llm_vs_actual') else '❌' }**")
    lines.append(f"- LLM vs Simulated: **{ '✅' if matches.get('llm_vs_simulated') else '❌' }**")
    lines.append("")

    if vsum:
        lines.append("## Discussion Fetch")
        kind = vsum.get("discussion_forum_kind") or "none"
        if kind != "none":
            lines.append(f"- Forum kind: {kind}")
            if vsum.get("discussion_url"):
                lines.append(f"- Discussion URL: {vsum.get('discussion_url')}")
            if kind == "discourse":
                if "discussion_posts_returned_total" in vsum:
                    lines.append(f"- Posts fetched (before cutoff): {vsum.get('discussion_posts_returned_total')}")
                if "discussion_posts_used_for_llm" in vsum:
                    lines.append(f"- Posts used for LLM: {vsum.get('discussion_posts_used_for_llm')}")
            elif kind == "generic":
                if "discussion_generic_status" in vsum:
                    lines.append(f"- HTTP status: {vsum.get('discussion_generic_status')}")
                if "discussion_generic_content_len" in vsum:
                    lines.append(f"- Content length: {vsum.get('discussion_generic_content_len')}")
        else:
            lines.append("- No linked forum discussion.")
        lines.append("")

    if resp.get("vote_summary"):
        lines.append("## Vote Timeline Summary (no final tally)")
        lines.append("```json")
        lines.append(json.dumps(resp["vote_summary"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    if resp.get("simulated_result"):
        lines.append("## Simulated Scores")
        lines.append("```json")
        lines.append(json.dumps(resp["simulated_result"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    if resp.get("actual_result"):
        lines.append("## Actual Result")
        lines.append("```json")
        lines.append(json.dumps(resp["actual_result"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    if resp.get("llm_decision"):
        lines.append("## LLM Decision")
        lines.append("```json")
        lines.append(json.dumps(resp["llm_decision"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        
        refs = resp.get("references_used") or []
        if refs:
            lines.append("## References (from Semantics)")
            for r in refs[:10]:  # show up to 10
                title = r.get("title") or "Untitled"
                year = r.get("year") or "n/a"
                url = r.get("url") or ""
                role = r.get("role") or "recent"
                lines.append(f"- [{role}] {title} ({year}) — {url}")
            lines.append("")

    # --- NEW: semantics section
    sem = resp.get("semantics_summary") or {}
    if sem:
        lines.append("## Semantics (research)")
        # show what we can in a compact way
        if resp.get("semantics_note"):
            lines.append(f"- Note: {resp.get('semantics_note')}")
        if resp.get("semantics_prompt_preview"):
            lines.append("### Semantics Prompt (preview)")
            lines.append("```text")
            lines.append(resp.get("semantics_prompt_preview") or "")
            lines.append("```")
        if sem.get("focal_meta"):
            lines.append("### Focal Paper")
            lines.append("```json")
            lines.append(json.dumps(sem.get("focal_meta"), ensure_ascii=False, indent=2))
            lines.append("```")
        if sem.get("recent_hits"):
            lines.append("### Recent Papers (trimmed)")
            lines.append("```json")
            lines.append(json.dumps(sem.get("recent_hits")[:5], ensure_ascii=False, indent=2))  # keep small
            lines.append("```")
        if resp.get("semantics_llm_json"):
            lines.append("### Semantics LLM JSON")
            lines.append("```json")
            lines.append(json.dumps(resp.get("semantics_llm_json"), ensure_ascii=False, indent=2))
            lines.append("```")
        elif resp.get("semantics_llm_response"):
            lines.append("### Semantics LLM Response")
            lines.append("```text")
            lines.append(resp.get("semantics_llm_response") or "")
            lines.append("```")
        lines.append("")

    compare_md = "\n".join(lines)
    with open(str(run_base) + "__compare.md", "w", encoding="utf-8") as f:
        f.write(compare_md)

    if resp.get("llm_prompt_preview"):
        clean = _clean_utf8(resp["llm_prompt_preview"])
        with open(str(run_base) + "__prompt_ascii.txt", "w", encoding="ascii", errors="ignore") as f:
            f.write(clean)

    if resp.get("llm_response"):
        clean = _clean_utf8(resp["llm_response"])
        with open(str(run_base) + "__llm_response.txt", "w", encoding="utf-8") as f:
            f.write(clean)

    print("\n✅ Done! Saved files:")
    print(" -", str(run_base) + "__response.json")
    print(" -", str(run_base) + "__compare.md")
    print(" -", str(run_base) + "__prompt_ascii.txt")
    if resp.get("llm_response"):
        print(" -", str(run_base) + "__llm_response.txt")
    pages_disp = "all" if int(answers['pages']) <= 0 else answers['pages']
    print(f"\nSettings: pages={pages_disp}, keep={answers['keep']}, body_max={answers['body_max']}, call_semantics={answers['call_semantics']}")
    print("\nChoices:", resp.get("choices"))
    print("Discussion URL:", resp.get("discussion_url"))
    print("Vote summary keys:", list((resp.get("vote_summary") or {}).keys()))
    if resp.get("semantics_summary"):
        print("Semantics included ✓")

if __name__ == "__main__":
    main()
