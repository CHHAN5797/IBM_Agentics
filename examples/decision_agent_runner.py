from __future__ import annotations

"""Utility helpers to run the governance ProposalDecision agent.

This module keeps the AI agent concerns separate from the larger example
script so that readers can focus on what the agent receives and returns.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agentics import Agentics as AG
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """Evidence item stored inside a ProposalDecision."""

    source_tool: str
    reference: Optional[str] = None
    quote: Optional[str] = None


class ActualVoteResult(BaseModel):
    """Stores the on-chain result to compare against the agent's prediction."""

    winner_label: Optional[str] = None
    winner_index: Optional[int] = None
    scores: Optional[List[float]] = None
    scores_total: Optional[float] = None
    margin_abs: Optional[float] = None
    margin_pct: Optional[float] = None


class ProposalDecision(BaseModel):
    """Outcome object produced by the AI agent."""

    snapshot_url: str
    selected_choice_label: str
    selected_choice_index: Optional[int] = None
    confidence: float
    summary: str
    key_arguments_for: List[str] = Field(default_factory=list)
    key_arguments_against: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    available_choices: Optional[List[str]] = None
    event_start_utc: Optional[str] = None
    event_end_utc: Optional[str] = None
    event_time_utc: Optional[str] = None
    address_of_governance_token: Optional[str] = None
    token_price_impact_pct: Optional[float] = None
    tvl_impact_pct: Optional[float] = None
    actual_vote_result: Optional[ActualVoteResult] = None
    simulation_reason: Optional[str] = None
    references: Optional[List[Dict[str, Any]]] = None
    ai_final_conclusion: Optional[str] = None
    ai_final_reason: Optional[str] = None


@dataclass
class DecisionAgentContext:
    """All inputs that shape the agent prompt."""

    snapshot_url: str
    choices: List[str]
    discussion_url: Optional[str]
    event_start_utc: Optional[str]
    event_end_utc: Optional[str]
    votes_count: int
    timeline_metrics: Dict[str, Any]
    token_price_impact_pct: Optional[float]
    tvl_impact_pct: Optional[float]
    adjacent_analytics: List[Dict[str, Any]]
    focus: Optional[str] = None


@dataclass
class DecisionAgentOutput:
    """Result from running the agent."""

    decision: ProposalDecision
    raw_result: Any
    pretty_print: Optional[str]




def _build_decision_prompt(ctx: DecisionAgentContext) -> List[str]:
    prompt = [
        f"Snapshot proposal under review: {ctx.snapshot_url}",
        (
            "Authoritative context: "
            f"choices={ctx.choices}, discussion={ctx.discussion_url}, "
            f"event_start_utc={ctx.event_start_utc}, event_end_utc={ctx.event_end_utc}"
        ),
        f"VOTES: count={ctx.votes_count} (FULL array fetched via MCP and used in timeline).",
        f"TIMELINE_METRICS(current): {json.dumps(ctx.timeline_metrics, ensure_ascii=False)}",
        (
            "MARKET_IMPACTS(current): "
            f"price_pct={ctx.token_price_impact_pct}, tvl_pct={ctx.tvl_impact_pct}"
        ),
        (
            "ADJACENT_ANALYTICS(≤3): "
            f"{json.dumps(ctx.adjacent_analytics, ensure_ascii=False)}"
        ),
        (
            "Also set `ai_final_conclusion` (one sentence: chosen option and stance) and "
            "`ai_final_reason` (2–4 bullet points integrating votes/timeline/market/TVL/adjacent)."
        ),
        "Objective: Choose exactly one option from the proposal's `choices`.",
        "Fill every field of ProposalDecision. Do NOT use ex-post tally.",
        "ONCHAIN GOAL (optional if needed): total holders & top-100 concentration.",
        "ANCHOR: Use END timestamp as the event window for market/TVL.",
        "Do NOT include token price impact (ex-post) as a pro/for argument.",
    ]
    if ctx.focus:
        prompt.append(f"Extra emphasis: {ctx.focus}")
    return prompt


def run_decision_agent(
    *,
    tools: List[Any],
    llm_provider: Any,
    context: DecisionAgentContext,
) -> DecisionAgentOutput:
    """Execute the ProposalDecision agent and return its structured output."""

    decision_prompt = "\n".join(_build_decision_prompt(context))

    decision_agent = AG(
        atype=ProposalDecision,
        tools=tools,
        max_iter=14,
        verbose_agent=True,
        description="Governance vote recommendation for a Snapshot proposal (ex-post blind).",
        instructions=(
            "Return a ProposalDecision object. Use the provided CONTEXT (timeline metrics, adjacent analytics). "
            "Use available MCP tools to gather forum discussion and sentiment analysis if needed. "
            "Choose exactly one option from `choices` and set both label and index. "
            "Include the full `choices` in available_choices. "
            "Set event_start_utc and event_end_utc (copy end into event_time_utc). "
            "Set address_of_governance_token to the authoritative token address."
        ),
        llm=llm_provider,
    )

    raw_result = asyncio.run(decision_agent << [decision_prompt])

    pretty = None
    pretty_method = getattr(raw_result, "pretty_print", None)
    if callable(pretty_method):
        try:
            pretty = pretty_method()
        except Exception:
            pretty = None

    states = getattr(raw_result, "states", [])
    if not states:
        raise RuntimeError("Agent produced no decision state; cannot proceed.")

    decision: ProposalDecision = states[0]

    return DecisionAgentOutput(decision=decision, raw_result=raw_result, pretty_print=pretty)


__all__ = [
    "DecisionAgentContext",
    "DecisionAgentOutput",
    "Evidence",
    "ActualVoteResult",
    "ProposalDecision",
    "run_decision_agent",
]
