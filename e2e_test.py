"""
End-to-end smoke test:
- Uses Semantics-Only MCP to (optionally) find papers and build planning/interpretation prompts
- Simulates an LLM to produce a ResearchPlan JSON (you can plug your real LLM)
- Pulls one finished Snapshot proposal, fetches votes
- Computes timeline metrics via Timeline MCP
- Builds an interpretation prompt and returns a "report" object
- Generates BibTeX for references

Run:
    export PYTHONPATH=src
    # (optional) export SEMANTIC_SCHOLAR_API_KEY=sk-...
    python e2e_semantics_check.py

Notes:
- This script is intentionally self-contained and avoids running separate MCP servers.
- It imports the MCP modules as libraries and calls their functions directly.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional

# --- Import your MCP modules as libraries (not as servers) ---
from agentics.mcp import semantics_mcp as sem        # Semantics-only (papers/prompts/bibtex)
from agentics.mcp import snapshot_api as snap             # Snapshot GraphQL client/tools
from agentics.mcp import timeline_mcp as tl               # Timeline metrics

# --------------------------------------------------------------------------------------
# 0) Config for the test
# --------------------------------------------------------------------------------------
GOAL = (
    "Analyze DAO proposals to detect coordinated or whale-driven voting patterns "
    "(e.g., turnout spikes, early-lead persistence) and link them to post-proposal "
    "token price and TVL reactions."
)
Y_VARS = ["token_return_[-1d,+3d]", "TVL_change_[0,+7d]"]
DOMAINS = ["DAO governance", "market microstructure"]

# Semantic Scholar queries (set empty list for offline-safe run)
# If you set queries and also export SEMANTIC_SCHOLAR_API_KEY, you'll get real papers.
QUERIES: List[str] = [
    "DAO governance vote manipulation",
    "turnout dynamics blockchain",
    "whale influence voting microstructure"
] if bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()) else []

SPACE = "aavedao.eth"     # You can switch to another known Snapshot space
LIMIT = 1                 # How many finished proposals to fetch/analyze in this smoke test


# --------------------------------------------------------------------------------------
# 1) Optional: a pluggable LLM function (you can replace with your real LLM call)
# --------------------------------------------------------------------------------------
def call_llm_for_research_plan(plan_prompt_text: str) -> Dict[str, Any]:
    """
    Dummy LLM: returns a minimal ResearchPlan JSON. Replace with your real LLM.
    - In production, enforce a strict JSON schema (pydantic/jsonschema).
    """
    # This dummy plan names data needs only; the actual fetching/calculation is done below.
    return {
        "objectives": [
            "Detect and quantify early-lead persistence",
            "Identify whale-like bursts and stairwise accumulation"
        ],
        "metrics": [
            {
                "name": "ELP",
                "purpose": "Measure early-lead persistence",
                "formula": "lead_hits_q1 / total_lead_hits",
                "required_data": ["snapshot_votes", "proposal_meta"],
                "estimator": "ratio",
                "references": []
            },
            {
                "name": "SpikeFollow",
                "purpose": "Large-vote spikes and immediate follow-through for the same option",
                "formula": "spike_follow_support_ratio",
                "required_data": ["snapshot_votes", "proposal_meta"],
                "estimator": "timeline_mcp.analyze_timeline (derived field)",
                "references": []
            }
        ],
        "data_plan": {
            "requests": [
                {"name": "proposal_meta", "source": "snapshot", "needs": ["proposal_id"]},
                {"name": "snapshot_votes", "source": "snapshot", "needs": ["proposal_id"]}
            ]
        },
        "eval_plan": {
            "robustness": ["alternate time windows", "exclude top-1 whale", "subsample by time"],
            "notes": "Observational; no causal identification."
        }
    }


def call_llm_for_interpretation(interpret_prompt_text: str) -> Dict[str, Any]:
    """
    Dummy LLM: returns a compact interpretation JSON. Replace with your real LLM.
    """
    return {
        "overview": "We analyze timeline-only indices without using the final tally for scoring.",
        "interpretation": (
            "A higher early-lead ratio indicates persistence; a high spike-follow ratio suggests "
            "coordinated behavior or whale influence. Combine both for a risk signal."
        ),
        "limitations": (
            "Observational data may be confounded by off-chain coordination, social media exposure, "
            "and proposal salience. Results are sensitive to window choices."
        ),
        "next_steps": [
            "Run robustness with alternate cutoffs (20%, 30%)",
            "Exclude top-1 and top-5% voters to assess whale sensitivity",
            "Include forum sentiment and social signals as controls"
        ]
    }


# --------------------------------------------------------------------------------------
# 2) Helpers
# --------------------------------------------------------------------------------------
def pick_one_finished_proposal(space: str) -> Optional[Dict[str, Any]]:
    """
    Choose a single finished Snapshot proposal for analysis.
    """
    items = snap.list_finished_proposals(snap.ProposalsIn(space=space, limit=max(1, LIMIT))) or []
    return items[0] if items else None


def compute_timeline_metrics(meta: Dict[str, Any], votes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Use Timeline MCP to compute timeline-only metrics from votes.
    """
    start = int(meta.get("start") or 0)
    end = int(meta.get("end") or 0)
    choices = meta.get("choices") or []
    return tl.analyze_timeline(start=start, end=end, choices=choices, votes=votes)


# --------------------------------------------------------------------------------------
# 3) Main E2E flow
# --------------------------------------------------------------------------------------
def main():
    print("== Health checks ==")
    print(json.dumps(sem.health(), indent=2))
    print(json.dumps(snap.health(), indent=2))
    print(json.dumps(tl.health(), indent=2))

    # (A) Build the planning prompt (optionally with papers)
    print("\n== Building plan prompt (Semantics) ==")
    plan_prompt_out = sem.plan_prompt(
        goal=GOAL,
        y_vars=Y_VARS,
        domains=DOMAINS,
        queries=QUERIES,
        per_query=1
    )
    print(f"papers_found = {len(plan_prompt_out.get('papers_found', []))}")
    print("(prompt preview) ", plan_prompt_out["prompt"][:300], "...")

    # (B) Use your LLM (dummy here) to produce a ResearchPlan JSON
    print("\n== Calling LLM for ResearchPlan (dummy) ==")
    research_plan = call_llm_for_research_plan(plan_prompt_out["prompt"])
    print(json.dumps(research_plan, indent=2))

    # (C) Fetch one finished proposal + votes from Snapshot according to the plan
    print("\n== Snapshot data fetch ==")
    prop = pick_one_finished_proposal(SPACE)
    if not prop:
        raise SystemExit(f"No finished proposals found in space={SPACE}")

    pid = prop["id"]
    meta = snap.get_proposal_by_id(pid)
    votes = snap.get_votes_all(pid, batch=500)

    print(f"Picked proposal_id={pid}, title={meta.get('title')}")
    print(f"votes fetched = {len(votes)}")

    # (D) Compute metrics with Timeline MCP
    print("\n== Timeline metrics ==")
    tl_metrics = compute_timeline_metrics(meta, votes)
    print(json.dumps({
        "lead_ratio_by_quartile": tl_metrics.get("lead_ratio_by_quartile"),
        "stability": tl_metrics.get("stability"),
        "early_ratio": tl_metrics.get("early_ratio"),
        "spike_index": tl_metrics.get("spike_index"),
        "spike_follow_support_ratio": tl_metrics.get("spike_follow_support_ratio"),
        "stairwise_ratio": tl_metrics.get("stairwise_ratio"),
        "recommended_index": tl_metrics.get("recommended_index"),
    }, indent=2))

    # (E) Prepare computed_metrics to feed into interpretation
    computed_metrics = [
        {"name": "ELP", "value": tl_metrics.get("early_ratio")},
        {"name": "SpikeFollow", "value": tl_metrics.get("spike_follow_support_ratio")}
    ]

    # (F) Build interpretation prompt and call LLM for interpretation
    print("\n== Interpretation prompt (Semantics) ==")
    interp_prompt_out = sem.interpretation_prompt(plan_json=research_plan, computed_metrics=computed_metrics)
    print("(prompt preview) ", interp_prompt_out["prompt"][:300], "...")

    print("\n== Calling LLM for Interpretation (dummy) ==")
    interpretation = call_llm_for_interpretation(interp_prompt_out["prompt"])
    print(json.dumps(interpretation, indent=2))

    # (G) Bibliography (if papers were found)
    if plan_prompt_out.get("papers_found"):
        print("\n== BibTeX from papers ==")
        bib = sem.bibtex_from_papers(plan_prompt_out["papers_found"])
        print(bib[:600], "...\n")
    else:
        print("\n(No papers_found; skip BibTeX)")

    # (H) Final stitched report (toy format)
    final_report = {
        "goal": GOAL,
        "space": SPACE,
        "proposal_id": pid,
        "proposal_title": meta.get("title"),
        "metrics_preview": {
            "early_ratio": tl_metrics.get("early_ratio"),
            "spike_follow_support_ratio": tl_metrics.get("spike_follow_support_ratio")
        },
        "interpretation": interpretation,
        "references_count": len(plan_prompt_out.get("papers_found", []))
    }

    print("\n== Final report (preview) ==")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
