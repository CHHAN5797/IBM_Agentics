#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for orchestra.py with live progress & concurrency.

- Live streaming stdout/stderr with per-task labels
- Concurrency (--jobs)
- Picks the *best* final JSON (prefers bundle with "results")
- tz-aware datetime; robust ORCHESTRA_ROOT resolution
- Saves per-task logs and final output JSON(s)

Usage:
  cd /Users/chunghyunhan/Projects/agentics/src/agentics/mcp
  python run_orchestra.py ...
"""

from __future__ import annotations
import os, json, subprocess, sys, datetime, argparse, importlib.util, threading
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------- paths & dirs ---------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

_env = os.environ.get("ORCHESTRA_ROOT")
if _env:
    ORCHESTRA_ROOT = _env if os.path.isabs(_env) else os.path.abspath(os.path.join(HERE, _env))
else:
    ORCHESTRA_ROOT = HERE

DATA_DIR   = os.environ.get("DATA_DIR", os.path.join(ORCHESTRA_ROOT, "data"))
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
MODELS_DIR = os.path.join(DATA_DIR, "models")
LOGS_DIR   = os.path.join(DATA_DIR, "logs")
for _d in (OUTPUT_DIR, MODELS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------- defaults ---------------------------
DEFAULTS = {
    "space": "aavedao.eth",
    "goal": "Prioritize DAO’s sustainable growth and broad community benefit over short-term gains.",
    "proposal_ids": [],
    "start": "2024-09-01",
    "end":   "2024-09-03",
    "decide_hours_before_end": 24,
    "pre_days": 7,
    "post_days": 7,
    "price_interval": "1d",
    "train_start": None,
    "train_end":   None,
    "model_path": os.path.join(MODELS_DIR, "aave_online_logreg.npz"),
    "load_model": True,
    "save_model": True,
    "train_extend": True,
    "retrain_if_older_than_days": 45,
    "yes_threshold": 0.55,
    "abstain_band": 0.05,
    "jobs": 1,
}

# --------------------------- helpers ---------------------------
_print_lock = threading.Lock()
def log(msg: str, label: Optional[str] = None):
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}" if label is None else f"[{ts}] [{label}] {msg}"
    with _print_lock:
        print(line, flush=True)

def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def _proposal_end_by_id(space: str, prop_id: str) -> Optional[int]:
    """Use snapshot_api.py to fetch proposal end time (ts)."""
    snap_path = os.path.join(ORCHESTRA_ROOT, "snapshot_api.py")
    SNP = _load_module(snap_path, "snapshot_api")
    try:
        p = SNP._fetch_proposal_by_id(prop_id)
        end_ts = int(p.get("end") or 0)
        sp = p.get("space", {}).get("id") if isinstance(p.get("space"), dict) else p.get("space")
        if sp and space and sp != space:
            log(f"warn: space mismatch: proposal space={sp}, expected={space}", label=prop_id[:8])
        return end_ts if end_ts > 0 else None
    except Exception:
        all_props = SNP._fetch_all_proposals(space)
        for p in all_props:
            if p.get("id") == prop_id:
                return int(p.get("end") or 0) or None
    return None

def _tight_range_around_end(end_ts: int, days_padding: int = 1) -> Tuple[str, str]:
    end_day = datetime.datetime.fromtimestamp(end_ts, datetime.timezone.utc).date()
    s = (end_day - datetime.timedelta(days=days_padding)).isoformat()
    e = (end_day + datetime.timedelta(days=days_padding)).isoformat()
    return s, e

def _outfile_name(prefix: str, space: str) -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_space = space.replace(".", "_")
    return os.path.join(OUTPUT_DIR, f"{prefix}_{safe_space}_{stamp}.json")

def _logfile_paths(label: str) -> Tuple[str, str]:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = f"{label}_{stamp}"
    return (os.path.join(LOGS_DIR, f"{base}.stdout.log"),
            os.path.join(LOGS_DIR, f"{base}.stderr.log"))

def _stream_orchestra(label: str, args: List[str]) -> Dict[str, Any]:
    """
    Run orchestra.py streaming stdout live, prefixing with [label].
    Returns the best parsed JSON: prioritizes the final bundle with "results".
    """
    cmd = [sys.executable, os.path.join(ORCHESTRA_ROOT, "orchestra.py")] + args
    log(f"launch: {' '.join(cmd)}", label=label)

    out_path, err_path = _logfile_paths(label)
    last_json: Optional[Dict[str, Any]] = None
    best_json: Optional[Dict[str, Any]] = None
    full_stdout_lines: List[str] = []
    full_stderr_lines: List[str] = []

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            s = line.rstrip("\n")
            full_stdout_lines.append(s)
            # Try to parse JSON
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    last_json = parsed
                    if ("results" in parsed and isinstance(parsed["results"], list)) or \
                       (parsed.get("space") and parsed.get("period")):
                        best_json = parsed
            except Exception:
                pass
            log(s, label=label)

        err = proc.stderr.read()
        if err:
            for ln in err.splitlines():
                full_stderr_lines.append(ln)
                log(f"stderr: {ln}", label=label)

        ret = proc.wait()

    # persist logs
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(full_stdout_lines))
        with open(err_path, "w", encoding="utf-8") as f:
            f.write("\n".join(full_stderr_lines))
        log(f"logs saved: stdout={out_path}, stderr={err_path}", label=label)
    except Exception as e:
        log(f"warn: failed to write logs: {e}", label=label)

    if ret != 0:
        raise RuntimeError(f"orchestra failed (exit={ret}). See logs: {out_path} / {err_path}")

    if best_json is not None:
        return best_json
    if last_json is not None:
        return last_json
    return {"raw_stdout": "\n".join(full_stdout_lines)}

# --------------------------- main runner ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=DEFAULTS["space"])
    ap.add_argument("--goal",  default=DEFAULTS["goal"])

    ap.add_argument("--proposal-id", action="append", dest="proposal_ids",
                    help="Specific proposal_id(s). Use multiple --proposal-id to add more.")
    ap.add_argument("--start", default=DEFAULTS["start"])
    ap.add_argument("--end",   default=DEFAULTS["end"])

    ap.add_argument("--decide-hours-before-end", type=int, default=DEFAULTS["decide_hours_before_end"])
    ap.add_argument("--pre", type=int, default=DEFAULTS["pre_days"])
    ap.add_argument("--post", type=int, default=DEFAULTS["post_days"])
    ap.add_argument("--price-interval", default=DEFAULTS["price_interval"], choices=("1d","1h"))

    ap.add_argument("--train-start", default=DEFAULTS["train_start"])
    ap.add_argument("--train-end",   default=DEFAULTS["train_end"])
    ap.add_argument("--model-path",  default=DEFAULTS["model_path"])
    ap.add_argument("--load-model",  action="store_true", default=DEFAULTS["load_model"])
    ap.add_argument("--save-model",  action="store_true", default=DEFAULTS["save_model"])
    ap.add_argument("--train-extend", action="store_true", default=DEFAULTS["train_extend"])
    ap.add_argument("--retrain-if-older-than-days", type=int, default=DEFAULTS["retrain_if_older_than_days"])

    ap.add_argument("--yes-threshold", type=float, default=DEFAULTS["yes_threshold"])
    ap.add_argument("--abstain-band",  type=float, default=DEFAULTS["abstain_band"])
    ap.add_argument("--jobs", type=int, default=DEFAULTS["jobs"], help="Parallel workers for proposal_ids mode")

    args = ap.parse_args()
    proposal_ids = args.proposal_ids or DEFAULTS["proposal_ids"]
    results: List[Dict[str, Any]] = []

    if proposal_ids:
        tasks: List[Tuple[str, List[str]]] = []
        for pid in proposal_ids:
            short = pid[:8]
            log("resolving end_ts...", label=short)
            end_ts = _proposal_end_by_id(args.space, pid)
            if not end_ts:
                log("skip: cannot resolve end_ts", label=short)
                continue
            s, e = _tight_range_around_end(end_ts, days_padding=1)
            run_args = [
                "--space", args.space,
                "--start", s, "--end", e,
                "--pre", str(args.pre), "--post", str(args.post),
                "--price-interval", args.price_interval,
                "--decide-hours-before-end", str(args.decide_hours_before_end),
                "--goal", args.goal,
                "--online-learn",  # keep online update default on
            ]
            if args.train_start and args.train_end:
                run_args += ["--train-start", args.train_start, "--train-end", args.train_end]
            if args.model_path: run_args += ["--model-path", args.model_path]
            if args.load_model: run_args += ["--load-model"]
            if args.save_model: run_args += ["--save-model"]
            if args.train_extend: run_args += ["--train-extend"]
            if args.retrain_if_older_than_days and args.retrain_if_older_than_days > 0:
                run_args += ["--retrain-if-older-than-days", str(args.retrain_if_older_than_days)]
            run_args += ["--yes-threshold", str(args.yes_threshold), "--abstain-band", str(args.abstain_band)]

            log(f"args ready: start={s}, end={e}, cutoff={args.decide_hours_before_end}h", label=short)
            tasks.append((short, run_args))

        if not tasks:
            log("no runnable tasks. exiting.")
            return

        jobs = max(1, int(args.jobs))
        log(f"starting {len(tasks)} task(s) with jobs={jobs}")

        futures = {}
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            for label, run_args in tasks:
                futures[ex.submit(_stream_orchestra, label, run_args)] = (label, run_args)
            for fut in as_completed(futures):
                label, run_args = futures[fut]
                try:
                    final_json = fut.result()
                    results.append({"label": label, "result": final_json})
                    log("completed.", label=label)
                except Exception as e:
                    log(f"FAILED: {e}", label=label)

        outfile = _outfile_name("orchestra_batch_proposals", args.space)
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump({"space": args.space, "proposals": results}, f, ensure_ascii=False, indent=2)
        log(f"saved: {outfile}")

    else:
        label = f"{args.space.split('.')[0]}-range"
        run_args = [
            "--space", args.space,
            "--start", args.start, "--end", args.end,
            "--pre", str(args.pre), "--post", str(args.post),
            "--price-interval", args.price_interval,
            "--decide-hours-before-end", str(args.decide_hours_before_end),
            "--goal", args.goal,
            "--online-learn",
        ]
        if args.train_start and args.train_end:
            run_args += ["--train-start", args.train_start, "--train-end", args.train_end]
        if args.model_path: run_args += ["--model-path", args.model_path]
        if args.load_model: run_args += ["--load-model"]
        if args.save_model: run_args += ["--save-model"]
        if args.train_extend: run_args += ["--train-extend"]
        if args.retrain_if_older_than_days and args.retrain_if_older_than_days > 0:
            run_args += ["--retrain-if-older-than-days", str(args.retrain_if_older_than_days)]
        run_args += ["--yes-threshold", str(args.yes_threshold), "--abstain-band", str(args.abstain_band)]

        log(f"range args ready: {args.start} ~ {args.end}, cutoff={args.decide_hours_before_end}h", label=label)
        final_json = _stream_orchestra(label, run_args)

        outfile = _outfile_name("orchestra_range", args.space)
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)
        log(f"saved: {outfile}", label=label)

if __name__ == "__main__":
    print(f"[runner] ORCHESTRA_ROOT = {ORCHESTRA_ROOT}")
    main()
