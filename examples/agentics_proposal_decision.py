"""Interactive Agentics example that issues a governance vote recommendation."""

from __future__ import annotations

import asyncio
import os
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from agentics import Agentics as AG
from agentics.core.llm_connections import get_llm_provider
from crewai_tools import MCPServerAdapter
from dotenv import load_dotenv
from mcp import StdioServerParameters
from pydantic import BaseModel, Field
import json


class Evidence(BaseModel):
    source_tool: str = Field(..., description="Name of the MCP tool that supplied the evidence.")
    reference: Optional[str] = Field(
        default=None,
        description="URL, identifier, or short pointer to the underlying resource, if available.",
    )
    quote: Optional[str] = Field(
        default=None,
        description="Short verbatim excerpt or data point copied from the MCP response.",
    )


class ProposalDecision(BaseModel):
    snapshot_url: str = Field(..., description="Snapshot proposal URL under review.")
    position: Literal["support", "oppose", "abstain"] = Field(
        ..., description="Recommended voting stance for the proposal (abstain when evidence is inconclusive)."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1 for the recommendation."
    )
    summary: str = Field(..., description="Concise narrative covering the final recommendation.")
    key_arguments_for: List[str] = Field(
        default_factory=list,
        description="Most compelling reasons to support the proposal (non-empty if recommending support).",
    )
    key_arguments_against: List[str] = Field(
        default_factory=list,
        description="Most compelling reasons to oppose the proposal (non-empty if recommending opposition).",
    )
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="Evidence items grounded in MCP tool outputs that justify the recommendation.",
    )
    risks: Optional[List[str]] = Field(
        default=None,
        description="Open risks, uncertainties, or missing data that could change the vote.",
    )
    follow_up_actions: Optional[List[str]] = Field(
        default=None,
        description="Concrete actions or verifications suggested before executing the vote.",
    )


def _server_params(script_path: Path, src_dir: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command="python3",
        args=[str(script_path)],
        env={"PYTHONPATH": str(src_dir), **os.environ},
    )


def _load_mcp_tools(project_root: Path, stack: ExitStack) -> List:
    src_dir = project_root / "src"
    server_specs = [
        ("snapshot", project_root / "src/agentics/mcp/snapshot_api.py"),
        ("timeline", project_root / "src/agentics/mcp/timeline_mcp.py"),
        ("forums", project_root / "src/agentics/mcp/forums_mcp.py"),
        ("govnews", project_root / "src/agentics/mcp/govnews_mcp.py"),
        ("defillama", project_root / "src/agentics/mcp/defillama_mcp.py"),
        ("holders", project_root / "src/agentics/mcp/holders_activity_mcp.py"),
        ("onchain", project_root / "src/agentics/mcp/onchain_activity_mcp_bq_cmc.py"),
        # ("cmc", project_root / "src/agentics/mcp/cmc_mcp.py"),
        ("cmc", project_root / "src/agentics/mcp/cmc_offline_mcp.py"),
    ]

    combined = None
    for name, script_path in server_specs:
        if not script_path.exists():
            print(f"[skip] {name}: missing server at {script_path}")
            continue
        adapter = MCPServerAdapter(_server_params(script_path, src_dir))
        try:
            tool_set = stack.enter_context(adapter)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            print(f"[skip] {name}: MCP server failed -> {exc}")
            continue
        print(f"Available {name} tools: {[tool.name for tool in tool_set]}")
        combined = tool_set if combined is None else combined + tool_set

    if combined is None:
        raise RuntimeError("No MCP servers could be started. Check dependencies and configuration.")

    return combined


def main() -> None:
    load_dotenv()
    project_root = Path(__file__).resolve().parent.parent

    with ExitStack() as stack:
        try:
            tools = _load_mcp_tools(project_root, stack)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

        try:
            llm = get_llm_provider()
        except ValueError as exc:
            raise SystemExit(
                "No LLM provider is configured. Populate your .env or environment variables before running."
            ) from exc

        snapshot_url = input("Snapshot Proposal URL> ").strip()
        if not snapshot_url:
            raise SystemExit("A Snapshot proposal URL is required to continue.")

        focus = input("Focus areas or concerns (optional)> ").strip()
        prompt = [
            f"Snapshot proposal under review: {snapshot_url}",
            "Objective: Determine whether to support, oppose, or abstain on the proposal.",
            "Requirements: Fill every field of the ProposalDecision schema.",
            "Use MCP tools to gather verifiable facts, votes, timelines, and market data.",
            "Cite tool outputs inside the evidence list and keep quotes faithful to the source.",
            "If data is insufficient, explain the gaps in risks or follow_up_actions.",
        ]
        if focus:
            prompt.append(f"Extra emphasis: {focus}")
        prompt_text = "\n".join(prompt)

        agent = AG(
            atype=ProposalDecision,
            tools=tools,
            max_iter=12,
            verbose_agent=True,
            description="Governance vote recommendation for a Snapshot proposal.",
            instructions=(
                "Return a ProposalDecision object. Set snapshot_url to the provided URL and choose "
                "position from 'support', 'oppose', or 'abstain' strictly based on gathered evidence."
            ),
            llm=llm,
        )

        result = asyncio.run(agent << [prompt_text])
        print(result.pretty_print())

        states = getattr(result, "states", [])
        if not states:
            raise SystemExit("Agent produced no decision state; cannot write log.")
        decision = states[0]
        payload = {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_url": snapshot_url,
            "focus": focus or None,
            "decision": decision.model_dump(mode="json"),
        }

        decision_dir = project_root / "Decision_runs"
        decision_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot_slug = snapshot_url.strip().split("/")[-1] or "proposal"
        outfile = decision_dir / f"decision_{snapshot_slug}_{timestamp}.json"
        with outfile.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"Saved structured decision to {outfile}")


if __name__ == "__main__":
    main()
