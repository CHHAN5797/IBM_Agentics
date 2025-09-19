"""
Quick check script for Semantics-Only MCP.
Run with:
    export PYTHONPATH=src
    python quickcheck_semantics_mcp.py
"""

import json
from agentics.mcp import semantics_mcp as sem

def main():
    # --- 1. Health check ---
    print("== Health ==")
    print(json.dumps(sem.health(), indent=2))

    # --- 2. Build a plan prompt (no queries -> offline safe) ---
    goal = "Detect coordinated voting patterns in DAO governance."
    y_vars = ["token_return_[-1d,+3d]", "TVL_change_[0,+7d]"]
    domains = ["DAO governance", "market microstructure"]

    plan_out = sem.plan_prompt(
        goal=goal,
        y_vars=y_vars,
        domains=domains,
        queries=[],         # leave empty to avoid network
        per_query=2
    )

    print("\n== Plan Prompt (truncated) ==")
    print(plan_out["prompt"][:600], "...\n")  # show only the first chunk of prompt text
    print(f"papers_found = {len(plan_out['papers_found'])}")

    # --- 3. Interpretation prompt with dummy metrics ---
    dummy_plan = {
        "objectives": ["Measure early-lead persistence"],
        "metrics": [
            {
                "name": "ELP",
                "purpose": "early lead persistence",
                "formula": "lead_hits_q1 / total_lead_hits",
                "required_data": ["votes"],
                "estimator": "ratio",
                "references": []
            }
        ],
        "data_plan": {"requests":[{"name":"votes","source":"snapshot","needs":["proposal_id"]}]},
        "eval_plan": {"robustness":["alt windows"], "notes":"observational"}
    }
    dummy_metrics = [{"name": "ELP", "value": 0.71, "notes": "q=25%"}]

    interp_out = sem.interpretation_prompt(plan_json=dummy_plan, computed_metrics=dummy_metrics)
    print("\n== Interpretation Prompt (truncated) ==")
    print(interp_out["prompt"][:600], "...\n")

if __name__ == "__main__":
    main()
