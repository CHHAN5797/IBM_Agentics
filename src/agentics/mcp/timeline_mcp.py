# src/agentics/mcp/timeline_mcp.py
from __future__ import annotations
"""
Vote timeline analyzer as a FastMCP stdio server.

Tool:
- analyze_timeline(start: int, end: int, choices: list[str], votes: list[dict]) -> dict
  * Timeline-only metrics (quartile lead ratios, stability, early_ratio, spike/stair, half-slope)
  * IMPORTANT: Does NOT use final tally for scoring; tally can be shown only for reporting.
"""
from typing import List, Dict, Any, Tuple
from statistics import mean
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("TimelineMCP")

def _compute_option_steps(votes_sorted: List[dict], k: int) -> Tuple[List[dict], List[float]]:
    cum_by_opt = [0.0] * k
    steps = []
    for v in votes_sorted:
        try:
            idx = int(v.get("choice")) - 1
        except Exception:
            idx = -1
        vp = float(v.get("vp") or 0.0)
        if 0 <= idx < k and vp > 0:
            cum_by_opt[idx] += vp
            steps.append({"t": int(v.get("created") or 0),
                          "voter": (v.get("voter") or "").lower().strip(),
                          "choice_idx": idx, "vp": vp})
    return steps, cum_by_opt

def _compute_spike_and_stair(steps: List[dict], winner_idx: int, winner_total_vp: float,
                             follow_window_votes: int = 30, follow_window_time_ratio: float = 0.25,
                             top_p_share: float = 0.10) -> Dict[str, float]:
    if not steps or winner_total_vp <= 0:
        return {"spike_index": 0.0, "spike_follow_support_ratio": 0.0, "stairwise_ratio": 0.0}
    wsteps = [s for s in steps if s["choice_idx"] == winner_idx]
    if not wsteps:
        return {"spike_index": 0.0, "spike_follow_support_ratio": 0.0, "stairwise_ratio": 0.0}

    max_step = max(wsteps, key=lambda s: s["vp"])
    spike_index = max_step["vp"] / winner_total_vp

    t_spike = max_step["t"]
    all_ts = [s["t"] for s in steps] or [t_spike]
    t_min, t_max = min(all_ts), max(all_ts)
    time_cut = t_spike + (t_max - t_min) * float(follow_window_time_ratio)

    after = [s for s in steps if s["t"] > t_spike][:follow_window_votes]
    after = [s for s in after if s["t"] <= time_cut]
    if not after:
        follow_ratio = 0.0
    else:
        vp_total = sum(s["vp"] for s in after)
        vp_same = sum(s["vp"] for s in after if s["choice_idx"] == winner_idx)
        follow_ratio = (vp_same / vp_total) if vp_total > 0 else 0.0

    sorted_steps = sorted((s["vp"] for s in wsteps), reverse=True)
    k = max(1, int(len(sorted_steps) * top_p_share))
    top_mass = sum(sorted_steps[:k])
    stairwise_ratio = 1.0 - (top_mass / winner_total_vp)

    return {"spike_index": round(spike_index, 6),
            "spike_follow_support_ratio": round(follow_ratio, 6),
            "stairwise_ratio": round(stairwise_ratio, 6)}

def _half_slope_diff(cum_series: List[Tuple[int, float]]) -> float:
    if len(cum_series) < 3: return 0.0
    n = len(cum_series); mid = n // 2
    def avg_slope(seg: List[Tuple[int, float]]) -> float:
        if len(seg) < 2: return 0.0
        slopes = []
        for (t1, y1), (t2, y2) in zip(seg[:-1], seg[1:]):
            dt = max(1, t2 - t1)
            slopes.append((y2 - y1) / dt)
        return mean(slopes) if slopes else 0.0
    early = cum_series[:mid]; late = cum_series[mid:]
    return round(avg_slope(late) - avg_slope(early), 6)

@mcp.tool()
def analyze_timeline(start: int, end: int, choices: List[str], votes: List[dict]) -> Dict[str, Any]:
    k = len(choices or [])
    if k == 0:
        return {"summary": "no choices", "recommended_index": None}

    T0, T1 = int(start or 0), int(end or 0)
    duration = max(1, T1 - T0)
    q1 = T0 + int(0.25 * duration)
    q2 = T0 + int(0.50 * duration)
    q3 = T0 + int(0.75 * duration)

    def q_of(ts: int) -> int:
        if ts <= q1: return 0
        if ts <= q2: return 1
        if ts <= q3: return 2
        return 3

    votes_sorted = sorted((votes or []), key=lambda v: int(v.get("created") or 0))

    cum_counts = [0] * k
    lead_hits_by_q = [[0]*k for _ in range(4)]
    lead_hits_total = [0] * k
    votes_in_q = [0, 0, 0, 0]
    vp_by_q = [[0.0]*k for _ in range(4)]
    last_leader = None
    leader_changes_after_75 = 0

    for v in votes_sorted:
        ts = int(v.get("created") or 0)
        if ts < T0 or ts > T1: continue
        try:
            idx = int((v.get("choice") or 0)) - 1
        except Exception:
            idx = -1
        vp = float(v.get("vp") or 0.0)
        if 0 <= idx < k:
            cum_counts[idx] += 1
            qi = q_of(ts)
            votes_in_q[qi] += 1
            vp_by_q[qi][idx] += vp
            max_val = max(cum_counts)
            leaders = [i for i, val in enumerate(cum_counts) if val == max_val]
            leader = leaders[0] if leaders else None
            if leader is not None:
                lead_hits_by_q[qi][leader] += 1
                lead_hits_total[leader] += 1
                if last_leader is not None and leader != last_leader and ts >= q3:
                    leader_changes_after_75 += 1
                last_leader = leader

    lead_ratio_by_q: List[List[float]] = []
    for qi in range(4):
        denom = max(1, votes_in_q[qi])
        lead_ratio_by_q.append([round(lead_hits_by_q[qi][i] / denom, 6) for i in range(k)])

    early_lead_hits = lead_hits_by_q[0][:]
    total_hits = max(1, sum(lead_hits_total))
    lead_ratio_total = [lead_hits_total[i] / total_hits for i in range(k)]
    penalty = min(1.0, 0.15 * leader_changes_after_75)
    stability = [max(0.0, r - penalty) for r in lead_ratio_total]
    sum_q1 = max(1, sum(early_lead_hits))
    early_ratio = [early_lead_hits[i] / sum_q1 for i in range(k)]

    steps, cum_by_opt = _compute_option_steps(votes_sorted, k)
    if sum(cum_by_opt) > 0:
        winner_idx = max(range(k), key=lambda i: cum_by_opt[i])
        winner_total = cum_by_opt[winner_idx]
        winner_cum = []
        running = 0.0
        for s in [s for s in steps if s["choice_idx"] == winner_idx]:
            running += s["vp"]
            winner_cum.append((s["t"], running))
        spike_stair = _compute_spike_and_stair(steps, winner_idx, winner_total)
        half_slope = _half_slope_diff(winner_cum)
    else:
        winner_idx, spike_stair, half_slope = None, {"spike_index":0.0,"spike_follow_support_ratio":0.0,"stairwise_ratio":0.0}, 0.0

    final_score = [0.7 * stability[i] + 0.3 * early_ratio[i] for i in range(k)]
    rec_idx = max(range(k), key=lambda i: final_score[i])

    return {
        "summary": "timeline-only analysis (no final tally)",
        "recommended_index": rec_idx,
        "lead_ratio_by_quartile": lead_ratio_by_q,
        "lead_ratio_total": lead_ratio_total,
        "stability": stability,
        "early_ratio": early_ratio,
        "vp_by_quartile": vp_by_q,
        "spike_index": spike_stair["spike_index"],
        "spike_follow_support_ratio": spike_stair["spike_follow_support_ratio"],
        "stairwise_ratio": spike_stair["stairwise_ratio"],
        "half_slope_diff": half_slope,
    }

@mcp.tool()
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "TimelineMCP"}

if __name__ == "__main__":
    mcp.run(transport="stdio")
