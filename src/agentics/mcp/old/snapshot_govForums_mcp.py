# src/agentics/mcp/snapshot_govForums_mcp.py
# FastAPI server: Snapshot ↔ Governance Forums analyzer (+ optional Semantics/Research call)
# Run:
#   export PYTHONPATH=src
#   python -m uvicorn agentics.mcp.snapshot_govForums_mcp:app --reload --port 8000 --app-dir src

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone
from urllib.parse import urlparse
import requests, re, time, random, json, os, unicodedata
from pathlib import Path
from statistics import mean

# Optional: load .env (OPENAI_API_KEY / OPENAI_MODEL / SEMANTICS_BASE_URL)
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

SNAPSHOT_API = "https://hub.snapshot.org/graphql"
TIMEOUT = 30

# === NEW: external Semantics (Semantic Scholar MCP) base URL ===
SEMANTICS_BASE_URL = os.getenv("SEMANTICS_BASE_URL", "http://localhost:8010")

# Known governance forums (mostly Discourse)
FORUM_HOSTS = {
    "governance.aave.com",
    "forum.arbitrum.foundation",
    "forum.decentraland.org",
    "forum.balancer.fi",
    "gov.curve.finance",
    "gov.1inch.io",
    "forum.aura.finance",
    "research.lido.fi",
    "gov.uniswap.org",
    "forum.ceg.vote",
}

# HTTP session
session = requests.Session()
session.headers.update({"User-Agent": "mcp-snapshot-govforums/1.1.0-spike+semantics"})

BASE_SLEEP  = 0.6
MAX_RETRIES = 5
BACKOFF_BASE = 1.7
JITTER = (0.1, 0.35)

def _sleep():
    time.sleep(BASE_SLEEP + random.uniform(*JITTER))

def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())

def _ts_to_iso(ts: int) -> str:
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except Exception:
        return ""

# ---------- Unicode clean helpers ----------
_ZW_CTRL_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]+|[\x00-\x08\x0B\x0C\x0E-\x1F]")

def _clean_utf8(s: Optional[str]) -> str:
    """Normalize to NFC and remove zero-width & control characters."""
    if not isinstance(s, str):
        return "" if s is None else str(s)
    s = unicodedata.normalize("NFC", s)
    s = _ZW_CTRL_RE.sub("", s)
    return s

# --- Utilities ---
def host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def is_discourse_thread(url: str) -> bool:
    h = host_of(url)
    path = urlparse(url).path.lower()
    return h in FORUM_HOSTS and "/t/" in path

def gql(query: str, variables: Optional[dict] = None) -> dict:
    """Call Snapshot GraphQL with retries on 429/5xx."""
    retries = 0
    while True:
        _sleep()
        try:
            r = session.post(SNAPSHOT_API, json={"query": query, "variables": variables or {}}, timeout=TIMEOUT)
        except requests.RequestException:
            if retries < MAX_RETRIES:
                delay = (BACKOFF_BASE ** retries) + random.uniform(*JITTER)
                time.sleep(delay)
                retries += 1
                continue
            raise
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 502, 503, 504) and retries < MAX_RETRIES:
            ra = r.headers.get("Retry-After")
            delay = float(ra) if (ra and ra.isdigit()) else (BACKOFF_BASE ** retries)
            time.sleep(delay + random.uniform(*JITTER))
            retries += 1
            continue
        r.raise_for_status()

def fetch_url(url: str) -> requests.Response:
    _sleep()
    return session.get(url, timeout=TIMEOUT)

def fetch_discourse_thread(url: str, max_pages: int = 5) -> Dict[str, Any]:
    """Discourse topic → .json + pagination."""
    base = url.split("?")[0].rstrip("/")
    if not base.endswith(".json"):
        base = base + ".json"
    posts, header = [], {}
    for page in range(1, max_pages + 1):
        u = base if page == 1 else base.replace(".json", f".json?page={page}")
        rr = fetch_url(u)
        if rr.status_code != 200:
            break
        j = rr.json()
        if page == 1:
            header = {
                "title": j.get("title"),
                "slug": j.get("slug"),
                "created_at": j.get("created_at"),
                "posts_count": j.get("posts_count"),
                "tags": j.get("tags"),
                "url": url,
            }
        chunk = j.get("post_stream", {}).get("posts", [])
        if not chunk:
            break
        for p in chunk:
            posts.append({
                "id": p.get("id"),
                "username": p.get("username"),
                "user_id": p.get("user_id"),
                "created_at": p.get("created_at"),
                "updated_at": p.get("updated_at"),
                "raw": p.get("raw"),
                "cooked": p.get("cooked"),
                "post_number": p.get("post_number"),
                "reply_to_post_number": p.get("reply_to_post_number"),
            })
    return {"type": "discourse", "thread": header, "posts": posts, "posts_returned": len(posts)}

def fetch_generic_page(url: str, max_bytes: int = 400_000) -> Dict[str, Any]:
    """If not Discourse, return partial HTML for context."""
    rr = fetch_url(url)
    text = rr.text if rr.status_code == 200 else ""
    if len(text) > max_bytes:
        text = text[:max_bytes]
    return {"type": "generic", "url": url, "status": rr.status_code, "content": text}

# --- Snapshot queries ---
PROPOSALS_Q = """
query($space: String!, $first: Int!, $skip: Int!) {
  proposals(
    first: $first
    skip: $skip
    where: { space_in: [$space] }
    orderBy: "created"
    orderDirection: desc
  ) {
    id
    title
    author
    body
    discussion
    start
    end
    state
  }
}
"""

PROPOSAL_BY_ID_Q = """
query($id: String!) {
  proposal(id: $id) {
    id
    title
    body
    author
    choices
    start
    end
    discussion
    state
  }
}
"""

PROPOSAL_RESULT_Q = """
query($id: String!) {
  proposal(id: $id) {
    id
    choices
    scores
    scores_total
    state
  }
}
"""

VOTES_BY_PROPOSAL_Q = """
query($proposal: String!, $first: Int!, $skip: Int!) {
  votes(
    first: $first
    skip: $skip
    where: { proposal: $proposal }
    orderBy: "created"
    orderDirection: asc
  ) {
    id
    voter
    created
    choice
    vp
    reason
  }
}
"""

def fetch_all_proposals(space: str, batch: int = 100) -> List[dict]:
    out, skip = [], 0
    while True:
        data = gql(PROPOSALS_Q, {"space": space, "first": batch, "skip": skip})
        chunk = (data.get("data") or {}).get("proposals") or []
        if not chunk:
            break
        out.extend(chunk)
        if len(chunk) < batch:
            break
        skip += batch
    return out

def finished_only(proposals: List[dict]) -> List[dict]:
    nt = _now_ts()
    return [p for p in proposals if p.get("state") == "closed" and int(p.get("end") or 0) <= nt]

def with_discussion_only(proposals: List[dict]) -> List[dict]:
    out = []
    for p in proposals:
        disc = (p.get("discussion") or "").strip()
        if disc:
            out.append(p)
    return out

# --- Text helpers ---
MD_CODE_BLOCK_RE = re.compile(r"```.+?```", re.S)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(MD_CODE_BLOCK_RE, " ", s)
    s = re.sub(HTML_TAG_RE, " ", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    s = re.sub(WS_RE, " ", s).strip()
    return _clean_utf8(s)

def norm_for_overlap(s: str) -> List[str]:
    s = clean_text(s).lower()
    s = re.sub(r"[^a-z0-9가-힣\s]", " ", s)
    toks = [t for t in s.split() if len(t) > 2]
    return toks

def is_mostly_contained(sub: str, full: str, jaccard_thresh: float = 0.75, contain_thresh: float = 0.8) -> bool:
    sub_t = norm_for_overlap(sub)
    full_t = norm_for_overlap(full)
    if not sub_t or not full_t:
        return False
    set_sub, set_full = set(sub_t), set(full_t)
    inter = len(set_sub & set_full)
    jacc = inter / max(1, len(set_sub | set_full))
    contain = inter / max(1, len(set_sub))
    long_enough = len(" ".join(sub_t)) >= 80
    return long_enough and (jacc >= jaccard_thresh or contain >= contain_thresh)

def truncate_chars(s: str, max_chars: int) -> str:
    s = _clean_utf8(s or "")
    return s if len(s) <= max_chars else (s[:max_chars] + " …")

def summarize_body_for_llm(title: str, body: str, max_chars: int = 3000) -> str:
    body_clean = clean_text(body)
    out = f"Title: {_clean_utf8(title)}\n\nBody:\n{body_clean}"
    return truncate_chars(out, max_chars=max_chars)

def summarize_comments_for_llm(comments: List[dict], end_ts: int, proposal_body: str,
                               max_keep: int = 20, max_chars_total: int = 6000) -> Tuple[str, List[Dict[str, Any]]]:
    def to_ts(iso: str) -> int:
        try:
            return int(datetime.fromisoformat(iso.replace("Z","+00:00")).timestamp())
        except Exception:
            return 0

    rows = []
    for p in comments:
        txt_raw = p.get("raw") or p.get("cooked") or ""
        txt = clean_text(txt_raw)
        if len(txt) < 15:
            continue
        ciso = p.get("created_at") or ""
        cts = to_ts(ciso)
        if end_ts and cts and cts > end_ts:
            continue
        if is_mostly_contained(txt, proposal_body):
            continue
        rows.append({"ts": cts, "username": p.get("username"), "text": txt})

    rows.sort(key=lambda r: (-len(r["text"]), r["ts"] or 0))
    rows = rows[:max_keep]

    out_lines, used = ["Discussion (selected comments):"], 0
    for r in rows:
        line = f"- [{_clean_utf8(r['username'] or '')}] {r['text']}"
        if used + len(line) + 1 > max_chars_total:
            break
        out_lines.append(line)
        used += len(line) + 1

    result = "\n".join(out_lines)
    return _clean_utf8(result), rows

# --- Votes helpers with quartiles + new spike/stair metrics ---

def fetch_proposal_by_id(pid: str) -> dict:
    data = gql(PROPOSAL_BY_ID_Q, {"id": pid})
    return (data.get("data") or {}).get("proposal") or {}

def fetch_proposal_result_by_id(pid: str) -> dict:
    data = gql(PROPOSAL_RESULT_Q, {"id": pid})
    return (data.get("data") or {}).get("proposal") or {}

def fetch_votes_all(pid: str, batch: int = 500) -> List[dict]:
    out, skip = [], 0
    while True:
        data = gql(VOTES_BY_PROPOSAL_Q, {"proposal": pid, "first": batch, "skip": skip})
        chunk = (data.get("data") or {}).get("votes") or []
        if not chunk:
            break
        out.extend(chunk)
        if len(chunk) < batch:
            break
        skip += batch
    return out

def _compute_option_steps(votes_sorted: List[dict], choices_count: int) -> Tuple[List[dict], List[float]]:
    """
    Build per-vote incremental VP for each option (ascending time).
    Returns:
        steps: list of dicts {"t": ts, "voter": addr, "choice_idx": idx, "vp": vp}
        cum_by_opt: final cumulative VP by option
    """
    cum_by_opt = [0.0] * choices_count
    steps = []
    for v in votes_sorted:
        try:
            idx = int(v.get("choice")) - 1
        except Exception:
            idx = -1
        vp = float(v.get("vp") or 0.0)
        if 0 <= idx < choices_count and vp > 0:
            cum_by_opt[idx] += vp
            steps.append({
                "t": int(v.get("created") or 0),
                "voter": (v.get("voter") or "").lower().strip(),
                "choice_idx": idx,
                "vp": vp,
            })
    return steps, cum_by_opt

def _compute_spike_and_stair_metrics(
    steps: List[dict],
    winner_idx: int,
    winner_total_vp: float,
    follow_window_votes: int = 30,
    follow_window_time_ratio: float = 0.25,
    top_p_share: float = 0.10
) -> Dict[str, float]:
    """
    Compute:
      - spike_index: max single-step VP (for winner) / winner_total_vp
      - spike_follow_support_ratio: after the biggest spike (winner), in the next window
        (by count/time), share of VP that continues to support the same (winner) option
      - stairwise_ratio: 1 - (sum of top p% winner-steps / winner_total_vp)
    """
    winner_steps = [s for s in steps if s["choice_idx"] == winner_idx]
    if not winner_steps or winner_total_vp <= 0:
        return {"spike_index": 0.0, "spike_follow_support_ratio": 0.0, "stairwise_ratio": 0.0}

    # 1) Spike index
    max_step = max(winner_steps, key=lambda s: s["vp"])
    spike_index = max_step["vp"] / winner_total_vp

    # 2) Follow-through window
    t_spike = max_step["t"]
    all_ts = [s["t"] for s in steps] or [t_spike]
    t_min, t_max = min(all_ts), max(all_ts)
    time_cut = t_spike + (t_max - t_min) * float(follow_window_time_ratio)

    after = [s for s in steps if s["t"] > t_spike]
    after = after[:follow_window_votes]
    after = [s for s in after if s["t"] <= time_cut]

    if not after:
        spike_follow_support_ratio = 0.0
    else:
        vp_total = sum(s["vp"] for s in after)
        vp_same  = sum(s["vp"] for s in after if s["choice_idx"] == winner_idx)
        spike_follow_support_ratio = (vp_same / vp_total) if vp_total > 0 else 0.0

    # 3) Stairwise ratio
    sorted_steps = sorted((s["vp"] for s in winner_steps), reverse=True)
    k = max(1, int(len(sorted_steps) * top_p_share))
    top_mass = sum(sorted_steps[:k])
    stairwise_ratio = 1.0 - (top_mass / winner_total_vp)

    return {
        "spike_index": round(spike_index, 6),
        "spike_follow_support_ratio": round(spike_follow_support_ratio, 6),
        "stairwise_ratio": round(stairwise_ratio, 6),
    }

def _compute_half_slope_diff(cum_series: List[Tuple[int, float]]) -> float:
    """
    cum_series: list of (t, cum_vp_for_winner) sampled monotonically by step time.
    Returns: half_slope_diff = mean_slope_late - mean_slope_early
    """
    if len(cum_series) < 3:
        return 0.0

    n = len(cum_series)
    mid = n // 2

    def avg_slope(seg: List[Tuple[int, float]]) -> float:
        if len(seg) < 2:
            return 0.0
        slopes = []
        for (t1, y1), (t2, y2) in zip(seg[:-1], seg[1:]):
            dt = max(1, t2 - t1)
            slopes.append((y2 - y1) / dt)
        return mean(slopes) if slopes else 0.0

    early = cum_series[:mid]
    late  = cum_series[mid:]
    return round(avg_slope(late) - avg_slope(early), 6)

def analyze_vote_timeline(start: int, end: int, choices: List[str], votes: List[dict]) -> Dict[str, Any]:
    """
    Lead dynamics by quartile (count-based hits/ratios) + stability/early_ratio + new spike/stair metrics.
    NOTE: We DO NOT use final tally to score; we only analyze the timeline structure.
    """
    k = len(choices or [])
    if k == 0:
        return {"summary": "no choices", "recommended_index": None}

    # Defensive casts
    T0, T1 = int(start or 0), int(end or 0)
    duration = max(1, T1 - T0)

    # Quartiles
    q1 = T0 + int(0.25 * duration)
    q2 = T0 + int(0.50 * duration)
    q3 = T0 + int(0.75 * duration)

    # Prepare cumulative COUNT lead hits (not VP) per incoming vote
    cum_counts = [0] * k
    lead_hits_by_quartile = [[0]*k for _ in range(4)]
    lead_hits_total = [0] * k
    votes_in_quartile = [0, 0, 0, 0]
    vp_by_quartile = [[0.0]*k for _ in range(4)]

    leader_changes_after_75 = 0
    last_leader = None

    def q_of(ts: int) -> int:
        if ts <= q1: return 0
        if ts <= q2: return 1
        if ts <= q3: return 2
        return 3

    # Ensure chronological order
    votes_sorted = sorted((votes or []), key=lambda v: int(v.get("created") or 0))

    # Walk votes to compute count-based lead hits + VP by quartile
    for v in votes_sorted:
        ts = int(v.get("created") or 0)
        if ts < T0 or ts > T1:
            continue
        try:
            idx = int((v.get("choice") or 0)) - 1
        except Exception:
            idx = -1
        vp = float(v.get("vp") or 0.0)

        if 0 <= idx < k:
            cum_counts[idx] += 1
            qi = q_of(ts)
            votes_in_quartile[qi] += 1
            vp_by_quartile[qi][idx] += vp

            max_val = max(cum_counts)
            leaders = [i for i, val in enumerate(cum_counts) if val == max_val]
            leader = leaders[0] if leaders else None

            if leader is not None:
                lead_hits_by_quartile[qi][leader] += 1
                lead_hits_total[leader] += 1

                if last_leader is not None and leader != last_leader and ts >= q3:
                    leader_changes_after_75 += 1
                last_leader = leader

    # Ratios based on lead hits
    lead_ratio_by_quartile: List[List[float]] = []
    for qi in range(4):
        denom = max(1, votes_in_quartile[qi])
        lead_ratio_by_quartile.append([round(lead_hits_by_quartile[qi][i] / denom, 6) for i in range(k)])

    early_lead_hits = lead_hits_by_quartile[0][:]  # Q1

    total_hits = max(1, sum(lead_hits_total))
    lead_ratio_total = [lead_hits_total[i] / total_hits for i in range(k)]
    penalty = min(1.0, 0.15 * leader_changes_after_75)
    stability = [max(0.0, r - penalty) for r in lead_ratio_total]

    sum_q1 = max(1, sum(early_lead_hits))
    early_ratio = [early_lead_hits[i] / sum_q1 for i in range(k)]

    final_score = [0.7 * stability[i] + 0.3 * early_ratio[i] for i in range(k)]
    sum_fs = max(1e-12, sum(final_score))
    final_score_pct = [round(fs / sum_fs, 6) for fs in final_score]
    rec_idx = final_score.index(max(final_score)) if final_score else None

    # Unique voters and total VP
    unique_voters = set()
    total_vp = 0.0
    for v in votes_sorted:
        ts = int(v.get("created") or 0)
        if ts < T0 or ts > T1:
            continue
        addr = (v.get("voter") or "").lower().strip()
        if addr:
            unique_voters.add(addr)
        try:
            total_vp += float(v.get("vp") or 0.0)
        except Exception:
            pass

    # Spike vs Stairwise + follow-through + half-slope
    steps, cum_by_opt = _compute_option_steps(votes_sorted, k)
    if rec_idx is None:
        rec_idx = int(max(range(k), key=lambda i: cum_by_opt[i])) if k > 0 else None

    winner_idx = rec_idx if isinstance(rec_idx, int) else None
    winner_total_vp = float(cum_by_opt[winner_idx]) if (winner_idx is not None and 0 <= winner_idx < k) else 0.0

    winner_cum_series: List[Tuple[int, float]] = []
    cum = 0.0
    for s in steps:
        if s["choice_idx"] == winner_idx:
            cum += s["vp"]
        winner_cum_series.append((s["t"], cum))

    spike_metrics = _compute_spike_and_stair_metrics(
        steps=steps,
        winner_idx=winner_idx if winner_idx is not None else 0,
        winner_total_vp=winner_total_vp,
        follow_window_votes=30,
        follow_window_time_ratio=0.25,
        top_p_share=0.10
    )
    half_slope_diff = _compute_half_slope_diff(winner_cum_series)

    def top_margin(arr: List[float]) -> float:
        ordered = sorted(arr, reverse=True)
        if len(ordered) >= 2:
            return ordered[0] - ordered[1]
        return ordered[0] if ordered else 0.0

    voting_margin_by_quartile = [round(top_margin(vp_by_quartile[q]), 6) for q in range(4)]

    summary = {
        "early_lead_hits": early_lead_hits,
        "lead_hits_by_quartile": lead_hits_by_quartile,
        "lead_hits_total": lead_hits_total,
        "lead_ratio_by_quartile": lead_ratio_by_quartile,
        "votes_in_quartile": votes_in_quartile,
        "leader_changes_after_75pct": leader_changes_after_75,
        "stability": [round(x, 6) for x in stability],
        "early_ratio": [round(x, 6) for x in early_ratio],
        "final_score": [round(x, 6) for x in final_score],
        "final_score_pct": final_score_pct,
        "voters_participated": len(unique_voters),
        "voting_power_total": round(total_vp, 6),
        "vp_by_quartile": [[round(v, 6) for v in row] for row in vp_by_quartile],
        "voting_margin_by_quartile": voting_margin_by_quartile,

        "winner_index_for_metrics_1based": (winner_idx + 1) if isinstance(winner_idx, int) else None,
        "winner_total_vp_for_metrics": round(winner_total_vp, 6),
        "spike_index": spike_metrics["spike_index"],
        "spike_follow_support_ratio": spike_metrics["spike_follow_support_ratio"],
        "stairwise_ratio": spike_metrics["stairwise_ratio"],
        "half_slope_diff": half_slope_diff,

        "interpretation": (
            "Prefer options with early & sustained support; penalize late flips. "
            "Diagnostics: spike_index, stairwise_ratio, spike_follow_support_ratio, half_slope_diff."
        )
    }

    return {"summary": summary, "recommended_index": rec_idx}

# --- LLM prompt (English-only policy + cleaned text) ---
def build_llm_prompt(prop: dict, vote_ana: dict, comments_text: str, body_trimmed: str) -> str:
    lang_policy = (
        "Language policy:\n"
        "- Use ENGLISH ONLY for every part of the output.\n"
        "- If the source text is not in English, summarize/translate it into concise English.\n"
        "- Do NOT include non-English text in the output.\n"
    )
    chs = prop.get("choices") or []
    eval_criteria = (
        "- Protocol risk & safety (security, liquidity, market impact)\n"
        "- Cost/benefit & resource needs\n"
        "- Feasibility & implementation complexity/timeline\n"
        "- Governance precedent & alignment with prior decisions\n"
        "- Stakeholder consensus from discussion (quality of arguments, not volume)\n"
        "- Early support & consensus stability (prefer early sustained support; penalize late flips)\n"
        "- Spike vs. stairwise dynamics and follow-through (avoid isolated spikes without later support)"
    )

    prompt = f"""
{lang_policy}
You are a governance analyst. Select ONE proposal option that most benefits the protocol/organization.

Constraints:
- Do NOT use or infer the final tally/result. Use only the voting timeline dynamics and discussion content.
- Prefer options with early, sustained support; penalize late flips and isolated spikes without follow-through.

Proposal:
Title: {_clean_utf8(prop.get("title") or "")}
Author: {_clean_utf8(prop.get("author") or "")}
Choices: {", ".join(_clean_utf8(str(c)) for c in chs)}

{_clean_utf8(body_trimmed)}

Voting timeline analysis (no final tally):
{json.dumps(vote_ana.get("summary"), ensure_ascii=False, indent=2)}

{_clean_utf8(comments_text)}

Evaluation criteria:
{eval_criteria}

Output:
Return ONLY a JSON object with the following fields (no extra text):
{{
  "body_summary": ["<bullet1>", "<bullet2>", "<bullet3>", "<bullet4>", "<bullet5>"],
  "comment_sentiment": {{
    "positive": <int>,
    "negative": <int>,
    "neutral": <int>,
    "themes_positive": ["<short theme>", "..."],
    "themes_negative": ["<short theme>", "..."]
  }},
  "decision": {{
    "choice_index": <1-based index>,
    "choice_label": "<label>",
    "rationale_bullets": ["<1-2 lines each>"],
    "rationale_paragraph": "<one concise paragraph>",
    "confidence_0_1": <float between 0 and 1>
  }}
}}
"""
    return _clean_utf8(prompt)

def call_openai_chat(prompt: str, model: str = None) -> str:
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[DRY-RUN] No OPENAI_API_KEY set. Prompt prepared but not sent."
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        j = r.json()
        return (j.get("choices") or [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        return f"[ERROR calling OpenAI] {repr(e)}"

def parse_llm_json(text: Optional[str]) -> Optional[dict]:
    if not text or not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end+1])
    except Exception:
        return None
    return None

def parse_snapshot_url(url: str) -> Optional[str]:
    try:
        path = urlparse(url).path.lower()
        frag = urlparse(url).fragment.lower()
        target = frag if "proposal/" in frag else path
        m = re.search(r"proposal/([0-9a-zA-Z]+)", target)
        return m.group(1) if m else None
    except Exception:
        return None

# --- API Schemas ---
class SpacesIn(BaseModel):
    spaces: List[str] = Field(..., description="Snapshot space ids")

class ProposalOut(BaseModel):
    space: str
    id: str
    title: str
    author: Optional[str] = None
    end: int
    end_iso: str
    discussion_url: str
    forum_kind: str   # discourse | generic

class FetchForumIn(BaseModel):
    url: str
    end_ts_filter: Optional[int] = Field(None, description="Only comments at/before this UNIX timestamp")
    max_pages: Optional[int] = Field(5, description="Max Discourse pages to fetch")
    max_posts: Optional[int] = Field(None, description="Cap returned posts per request")

class FetchCommentsIn(BaseModel):
    spaces: List[str] = Field(..., description="Snapshot space ids")
    max_pages: Optional[int] = Field(5, description="Max Discourse pages to fetch")
    max_posts: Optional[int] = Field(None, description="Cap returned posts per proposal")
    use_proposal_end_as_cutoff: bool = Field(True, description="If True, include only comments before proposal end")

class AnalyzeIn(BaseModel): ### class names (string, type)
    '''
    docstring for AnalyzeIn
    '''
    snapshot_url: str = Field(..., description="Snapshot proposal URL")
    
    max_comment_pages: Optional[int] = Field(6, description="Discourse pagination depth (<=0 means no limit)")
    max_comment_keep: Optional[int] = Field(20, description="How many comments to pass to the LLM")
    body_max_chars: Optional[int] = Field(3000, description="Max characters of proposal body for LLM")
    call_llm: bool = Field(False, description="If True, call OpenAI API (OPENAI_API_KEY required)")


    # === NEW: semantics MCP call flags/params ===
    call_semantics: bool = Field(False, description="If True, call Semantic Scholar MCP and attach research mapping")
    semantics_prompt_mode: Optional[str] = Field("build", description="'build' or 'call' (call=ask OpenAI inside semantics MCP)")
    semantics_limit_recent: Optional[int] = Field(12, description="Recent papers to pull")
    semantics_year_from: Optional[int] = Field(2021)
    semantics_year_to: Optional[int] = Field(2025)
    semantics_focal_doi: Optional[str] = Field("10.2139/ssrn.4367209")
    semantics_topic: Optional[str] = Field("Decentralized Governance; DAO; on-chain voting; delegation; quadratic voting")

class AnalyzeOut(BaseModel):
    ok: bool
    proposal_id: Optional[str] = None
    proposal_title: Optional[str] = None
    choices: Optional[List[str]] = None
    discussion_url: Optional[str] = None
    vote_summary: Optional[Dict[str, Any]] = None
    body_summary: Optional[List[str]] = None
    comment_sentiment: Optional[Dict[str, Any]] = None
    llm_decision: Optional[Dict[str, Any]] = None
    actual_result: Optional[Dict[str, Any]] = None
    simulated_result: Optional[Dict[str, Any]] = None
    comparison: Optional[Dict[str, Any]] = None
    llm_prompt_preview: Optional[str] = None
    llm_response: Optional[str] = None
    notes: Optional[str] = None
    references_used: Optional[List[Dict[str, Any]]] = None 
    

    # === NEW: semantics result bundle ===
    semantics_summary: Optional[Dict[str, Any]] = None
    semantics_prompt_preview: Optional[str] = None
    semantics_llm_response: Optional[str] = None
    semantics_llm_json: Optional[Dict[str, Any]] = None
    semantics_note: Optional[str] = None

class ProposalWithCommentsOut(BaseModel):
    space: str 
    id: str
    title: str
    end: int
    end_iso: str
    discussion_url: str
    forum_kind: str
    posts_returned: int
    thread: Optional[Dict[str, Any]] = None
    posts: Optional[List[Dict[str, Any]]] = None
    generic_status: Optional[int] = None
    generic_content_len: Optional[int] = None

# --- FastAPI (MCP) ---
app = FastAPI(title="MCP: Snapshot ↔ Governance Forums", version="1.2.0-spike+semantics")

@app.get("/health")
def health():
    return {
        "ok": True,
        "ts": datetime.utcnow().isoformat(),
        "semantics_base": SEMANTICS_BASE_URL,
        "keys_loaded": {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "openai_model": os.getenv("OPENAI_MODEL") or None
        }
    }

@app.post("/snapshot/finished-proposals")
def finished_proposals(body: SpacesIn):
    out = []
    for space in body.spaces:
        props = finished_only(fetch_all_proposals(space))
        for p in props:
            disc = (p.get("discussion") or "").strip()
            url = disc if disc else None
            kind = "discourse" if (url and is_discourse_thread(url)) else ("generic" if url else "none")
            out.append({
                "space": space,
                "id": p["id"],
                "title": _clean_utf8(p.get("title", "")),
                "author": _clean_utf8(p.get("author") or ""),
                "end": int(p.get("end") or 0),
                "end_iso": _ts_to_iso(p.get("end") or 0),
                "discussion_url": url,
                "forum_kind": kind,
                "confidence": 1.0 if url else 0.0,
            })
    return out

@app.post("/snapshot/with-discussion/list", response_model=List[ProposalOut])
def snapshot_with_discussion_list(body: SpacesIn):
    out: List[ProposalOut] = []
    for space in body.spaces:
        props = finished_only(fetch_all_proposals(space))
        props = with_discussion_only(props)
        for p in props:
            url = (p.get("discussion") or "").strip()
            kind = "discourse" if is_discourse_thread(url) else "generic"
            out.append(ProposalOut(
                space=space,
                id=p["id"],
                title=_clean_utf8(p.get("title","")),
                author=_clean_utf8(p.get("author") or ""),
                end=int(p.get("end") or 0),
                end_iso=_ts_to_iso(p.get("end") or 0),
                discussion_url=url,
                forum_kind=kind,
            ))
    return out

@app.post("/snapshot/with-discussion/fetch-comments", response_model=List[ProposalWithCommentsOut])
def snapshot_with_discussion_fetch_comments(body: FetchCommentsIn):
    results: List[ProposalWithCommentsOut] = []
    for space in body.spaces:
        props = finished_only(fetch_all_proposals(space))
        props = with_discussion_only(props)
        for p in props:
            url = (p.get("discussion") or "").strip()
            kind = "discourse" if is_discourse_thread(url) else "generic"
            end_ts = int(p.get("end") or 0)

            mp = int(body.max_pages or 5)
            if mp <= 0:
                mp = 9999

            if kind == "discourse":
                data = fetch_discourse_thread(url, max_pages=mp)
                posts = data.get("posts", [])
                if body.use_proposal_end_as_cutoff and end_ts:
                    def to_ts(iso: str) -> int:
                        try:
                            return int(datetime.fromisoformat(iso.replace("Z","+00:00")).timestamp())
                        except Exception:
                            return 0
                    posts = [pp for pp in posts if pp.get("created_at") and to_ts(pp["created_at"]) <= end_ts]
                if body.max_posts is not None:
                    posts = posts[: int(body.max_posts)]
                results.append(ProposalWithCommentsOut(
                    space=space,
                    id=p["id"],
                    title=_clean_utf8(p.get("title","")),
                    end=end_ts,
                    end_iso=_ts_to_iso(end_ts),
                    discussion_url=url,
                    forum_kind=kind,
                    posts_returned=len(posts),
                    thread=data.get("thread"),
                    posts=posts
                ))
            else:
                g = fetch_generic_page(url, max_bytes=200_000)
                results.append(ProposalWithCommentsOut(
                    space=space,
                    id=p["id"],
                    title=_clean_utf8(p.get("title","")),
                    end=end_ts,
                    end_iso=_ts_to_iso(end_ts),
                    discussion_url=url,
                    forum_kind=kind,
                    posts_returned=0,
                    generic_status=g.get("status"),
                    generic_content_len=len(g.get("content") or "")
                ))
    return results

@app.post("/forums/fetch")
def forums_fetch(body: FetchForumIn):
    url = body.url.strip()
    mp = int(body.max_pages or 5)
    if mp <= 0:
        mp = 9999
    data = fetch_discourse_thread(url, max_pages=mp) if is_discourse_thread(url) else fetch_generic_page(url)
    if body.end_ts_filter and data.get("type") == "discourse":
        cutoff = int(body.end_ts_filter)
        def to_ts(iso: str) -> int:
            try:
                return int(datetime.fromisoformat(iso.replace("Z","+00:00")).timestamp())
            except Exception:
                return 0
        posts = [p for p in data.get("posts", []) if p.get("created_at") and to_ts(p["created_at"]) <= cutoff]
        if body.max_posts is not None:
            posts = posts[: int(body.max_posts)]
        data["posts"] = posts
        data["posts_returned"] = len(posts)
    elif data.get("type") == "discourse" and body.max_posts is not None:
        data["posts"] = data.get("posts", [])[: int(body.max_posts)]
        data["posts_returned"] = len(data["posts"])
    return data

# === NEW: tiny client for Semantics MCP HTTP ===
def call_semantics_mcp(
    topic: str,
    focal_doi: Optional[str],
    year_from: int,
    year_to: int,
    limit_recent: int,
    prompt_mode: str = "build",
    base_url: Optional[str] = None,
    timeout: int = 120
) -> Dict[str, Any]:
    """
    Call the Semantic Scholar MCP HTTP wrapper (/research_dao_governance).
    Returns JSON with: focal_meta, recent_hits, prompt, (optional) llm_response/llm_json, note.
    """
    base = (base_url or SEMANTICS_BASE_URL or "").rstrip("/")
    url = f"{base}/research_dao_governance"
    params = {
        "topic": topic,
        "focal_doi": focal_doi,
        "year_from": year_from, "year_to": year_to,
        "limit_recent": limit_recent,
        "prompt_mode": prompt_mode
    }
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": f"[semantics call failed] {e!r}"}

# ================== ANALYZE FROM SNAPSHOT URL ==================
@app.post("/snapshot/analyze-from-url", response_model=AnalyzeOut)
def analyze_from_url(body: AnalyzeIn)-> str: ## 
    """
    Steps:
      1) Fetch proposal & votes (ordered by time). Do NOT use final tally for decision.
      2) If discussion exists, fetch comments; compute fetch stats; prepare comment summary for LLM.
      3) Analyze timeline (quartiles + spike/stair metrics). Build summary.
      4) Optionally call LLM; fetch actual final tally for reporting only.
      5) (NEW) Optionally call Semantics MCP and attach research mapping.
    """
    def _parse_id(url: str) -> Optional[str]:
        try:
            path = urlparse(url).path.lower()
            frag = urlparse(url).fragment.lower()
            target = frag if "proposal/" in frag else path
            m = re.search(r"proposal/([0-9a-zA-Z]+)", target)
            return m.group(1) if m else None
        except Exception:
            return None

    pid = _parse_id(body.snapshot_url or "")
    if not pid:
        return AnalyzeOut(ok=False, notes="Could not parse proposal id from the URL.")

    # (1) Proposal & votes
    prop = fetch_proposal_by_id(pid)
    if not prop or not prop.get("id"):
        return AnalyzeOut(ok=False, notes="Proposal not found.")
    votes = fetch_votes_all(pid, batch=500)

    # (2) Discussion → comments text + fetch stats
    disc_url = (prop.get("discussion") or "").strip()
    comments_text = "Discussion: (no linked forum)"
    filtered_rows: List[Dict[str, Any]] = []
    discussion_forum_kind = "none"
    discussion_posts_returned_total = None
    discussion_posts_used_for_llm = None
    discussion_generic_status = None
    discussion_generic_content_len = None

    mp = int(body.max_comment_pages or 6)
    if mp <= 0:
        mp = 9999

    if disc_url:
        if is_discourse_thread(disc_url):
            discussion_forum_kind = "discourse"
            d = fetch_discourse_thread(disc_url, max_pages=mp)
            posts_all = d.get("posts") or []
            discussion_posts_returned_total = len(posts_all)
            comments_text, filtered_rows = summarize_comments_for_llm(
                posts_all,
                int(prop.get("end") or 0),
                proposal_body=prop.get("body") or "",
                max_keep=int(body.max_comment_keep or 20),
                max_chars_total=6000
            )
            discussion_posts_used_for_llm = len(filtered_rows or [])
        else:
            discussion_forum_kind = "generic"
            g = fetch_generic_page(disc_url, max_bytes=150_000)
            comments_text = f"Discussion page fetched (generic). status={g.get('status')}, content_len={len(g.get('content') or '')}. No structured comments."
            discussion_generic_status = g.get("status")
            discussion_generic_content_len = len(g.get("content") or "")
        comments_text = _clean_utf8(comments_text)

    # (3) Timeline analysis (no final tally)
    body_trimmed = summarize_body_for_llm(prop.get("title") or "", prop.get("body") or "", max_chars=int(body.body_max_chars or 3000))
    vote_ana = analyze_vote_timeline(prop.get("start"), prop.get("end"), prop.get("choices") or [], votes)

    # Enrich summary with a proposal header and discussion stats
    psum = {
        "id": prop.get("id"),
        "title": _clean_utf8(prop.get("title") or ""),
        "author": _clean_utf8(prop.get("author") or ""),
        "start_iso": _ts_to_iso(prop.get("start") or 0),
        "end_iso": _ts_to_iso(prop.get("end") or 0),
        "choices": prop.get("choices") or []
    }
    if isinstance(vote_ana, dict) and isinstance(vote_ana.get("summary"), dict):
        vote_ana["summary"]["proposal_summary"] = psum
        vote_ana["summary"]["discussion_forum_kind"] = discussion_forum_kind
        vote_ana["summary"]["discussion_url"] = disc_url or ""
        if discussion_forum_kind == "discourse":
            vote_ana["summary"]["discussion_posts_returned_total"] = discussion_posts_returned_total
            vote_ana["summary"]["discussion_posts_used_for_llm"] = discussion_posts_used_for_llm
        elif discussion_forum_kind == "generic":
            vote_ana["summary"]["discussion_generic_status"] = discussion_generic_status
            vote_ana["summary"]["discussion_generic_content_len"] = discussion_generic_content_len

    # (4) LLM prompt and optional call
    prompt = build_llm_prompt(prop, vote_ana, comments_text, body_trimmed)
    llm_resp = None
    llm_json = None
    if body.call_llm:
        llm_resp = call_openai_chat(prompt)
        llm_resp = _clean_utf8(llm_resp)
        llm_json = parse_llm_json(llm_resp)

    body_summary = None
    comment_sentiment = None
    llm_decision = None
    if isinstance(llm_json, dict):
        body_summary = llm_json.get("body_summary")
        comment_sentiment = llm_json.get("comment_sentiment")
        llm_decision = llm_json.get("decision")

    # (5) Actual final tally (reporting only)
    res = fetch_proposal_result_by_id(pid)
    actual = None
    if res and res.get("scores"):
        scores = res.get("scores") or []
        choices = res.get("choices") or prop.get("choices") or []
        if scores and choices and len(scores) == len(choices):
            max_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else None
            sorted_scores = sorted(scores, reverse=True)
            voting_margin_overall = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else (sorted_scores[0] if sorted_scores else 0.0)
            actual = {
                "state": res.get("state"),
                "choices": choices,
                "scores": scores,
                "scores_total": res.get("scores_total"),
                "winner_index": (max_idx + 1) if max_idx is not None else None,
                "winner_label": choices[max_idx] if max_idx is not None else None,
                "voting_margin_overall": voting_margin_overall
            }

    # (6) Simulated result (from timeline)
    choices_list = prop.get("choices") or []
    simulated = None
    try:
        vsum = vote_ana.get("summary") if isinstance(vote_ana, dict) else None
        sim_scores = vsum.get("final_score") if isinstance(vsum, dict) else None
        sim_scores_pct = vsum.get("final_score_pct") if isinstance(vsum, dict) else None
        rec_idx = vote_ana.get("recommended_index") if isinstance(vote_ana, dict) else None

        voting_margin_simulated = 0.0
        if isinstance(sim_scores, list) and sim_scores:
            ordered = sorted(sim_scores, reverse=True)
            voting_margin_simulated = (ordered[0] - ordered[1]) if len(ordered) >= 2 else ordered[0]

        if isinstance(sim_scores, list) and len(sim_scores) == len(choices_list):
            simulated = {
                "choices": choices_list,
                "scores": [round(s, 6) for s in sim_scores],
                "scores_pct": sim_scores_pct,
                "winner_index": (rec_idx + 1) if isinstance(rec_idx, int) else None,
                "winner_label": choices_list[rec_idx] if isinstance(rec_idx, int) and 0 <= rec_idx < len(choices_list) else None,
                "voting_margin_simulated": round(voting_margin_simulated, 6)
            }
    except Exception:
        simulated = None

    # (7) (NEW) Semantics MCP call (optional)
    semantics_payload = None
    semantics_prompt_preview = None
    semantics_llm_response = None
    semantics_llm_json = None
    semantics_note = None

    if body.call_semantics:
        semantics_payload = call_semantics_mcp(
            topic=body.semantics_topic or "Decentralized Governance; DAO; on-chain voting; delegation; quadratic voting",
            focal_doi=body.semantics_focal_doi,
            year_from=int(body.semantics_year_from or 2021),
            year_to=int(body.semantics_year_to or 2025),
            limit_recent=int(body.semantics_limit_recent or 12),
            prompt_mode=(body.semantics_prompt_mode or "build"),
            base_url=SEMANTICS_BASE_URL
        )
        if isinstance(semantics_payload, dict):
            semantics_prompt_preview = truncate_chars(semantics_payload.get("prompt") or "", 2200)
            semantics_llm_response = semantics_payload.get("llm_response")
            semantics_llm_json = semantics_payload.get("llm_json")
            semantics_note = semantics_payload.get("note") or semantics_payload.get("error")
            
            refs = []
            focal = semantics_payload.get("focal_meta")
            if focal:
                refs.append({
                    "role": "focal",
                    "title": focal.get("title"),
                    "year": focal.get("year"),
                    "url": focal.get("url")
                })
            for it in (semantics_payload.get("recent_hits") or []):
                refs.append({
                    "role": "recent",
                    "title": it.get("title"),
                    "year": it.get("year"),
                    "url": it.get("url")
                })
            references_used = refs or None

    # Build comparison
    def _safe_label(lst, idx1):
        try:
            return lst[idx1 - 1] if idx1 and 1 <= idx1 <= len(lst) else None
        except Exception:
            return None

    actual_winner_idx = (actual or {}).get("winner_index")
    actual_winner_lbl = (actual or {}).get("winner_label")
    simulated_winner_idx = (simulated or {}).get("winner_index")
    simulated_winner_lbl = (simulated or {}).get("winner_label")

    llm_choice_idx = None
    llm_choice_lbl = None
    if isinstance(llm_decision, dict):
        llm_choice_idx = llm_decision.get("choice_index")
        llm_choice_lbl = llm_decision.get("choice_label") or _safe_label(choices_list, llm_choice_idx)

    comparison = {
        "actual": {
            "winner_index": actual_winner_idx,
            "winner_label": actual_winner_lbl,
            "state": (actual or {}).get("state"),
        },
        "simulated": {
            "winner_index": simulated_winner_idx,
            "winner_label": simulated_winner_lbl,
        },
        "llm": {
            "choice_index": llm_choice_idx,
            "choice_label": llm_choice_lbl,
        },
        "matches": {
            "simulated_vs_actual": (bool(simulated_winner_idx) and simulated_winner_idx == actual_winner_idx),
            "llm_vs_actual": (bool(llm_choice_idx) and llm_choice_idx == actual_winner_idx),
            "llm_vs_simulated": (bool(llm_choice_idx) and bool(simulated_winner_idx) and llm_choice_idx == simulated_winner_idx),
        }
    }

    return AnalyzeOut(
        ok=True,
        proposal_id=prop.get("id"),
        proposal_title=_clean_utf8(prop.get("title") or ""),
        choices=prop.get("choices"),
        discussion_url=disc_url or None,
        vote_summary=vote_ana.get("summary"),
        body_summary=body_summary,
        comment_sentiment=comment_sentiment,
        llm_decision=llm_decision,
        actual_result=actual,
        simulated_result=simulated,
        comparison=comparison,
        llm_prompt_preview=truncate_chars(prompt, 2200),
        llm_response=llm_resp,
        notes=("Decision uses timeline stability and discussion; final tally is fetched after the decision solely for reporting."),

        # NEW: attach semantics
        semantics_summary=semantics_payload if isinstance(semantics_payload, dict) else None,
        semantics_prompt_preview=semantics_prompt_preview,
        semantics_llm_response=semantics_llm_response,
        semantics_llm_json=semantics_llm_json,
        semantics_note=semantics_note,
        references_used=references_used
    )


# build, call 이름 바꾸기 (call은 프롬프트와 LLM 호출까지 수행)
# call 잘 작동하는지 확인하기 
# 각 metric에 대한 설명 달기 (excel)