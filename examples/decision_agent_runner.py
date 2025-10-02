from __future__ import annotations

"""Utility helpers to run the governance ProposalDecision agent.

This module keeps the AI agent concerns separate from the larger example
script so that readers can focus on what the agent receives and returns.
"""

import asyncio
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agentics import Agentics as AG
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple


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

class SimilarProposalSummary(BaseModel):
    """Condensed analytics about related historical proposals."""

    proposal_id: str
    title: Optional[str] = None
    end_utc: Optional[str] = None
    summary: str
    similarity_reason: str
    market_response: str
    change_stance: Optional[str] = None

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
    decision_stance: Optional[str] = None
    ai_final_conclusion: Optional[str] = None
    ai_final_reason: Optional[str] = None
    similar_proposals: List[SimilarProposalSummary] = Field(default_factory=list)
    ex_post_price_impact_pct: Optional[float] = None
    ex_post_tvl_impact_pct: Optional[float] = None
    ex_post_window: Optional[str] = None
    ex_post_note: Optional[str] = None


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
                (
            "IMPACT REMINDER: Assume your recommendation could change the final tally; "
            "use vote progress as a data point but make an independent choice that "
            "maximizes the organization’s long-term health."
        ),
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
        "Also set `decision_stance` to exactly 'Change' or 'Not change, Status quo'; format "
            "`ai_final_conclusion` as `Option '<label>' selected with stance to keep '<stance>'.`; "
            "set `ai_final_reason` (2–4 bullet points integrating votes/timeline/market/TVL/adjacent)."
        ),
        (
            "HISTORICAL_LESSONS: retrieve similar past proposals, give 1–2 sentence summaries "
            "of what each attempted to change, and note whether post-vote token price or TVL "
            "declined (if so, treat that outcome as unsuccessful and learn from it)."
        ),
        (
            "Populate `similar_proposals` with exactly two SimilarProposalSummary entries using "
            "ADJACENT_ANALYTICS. Provide summaries, similarity reasons, market responses, and "
            "change stances even if some fields must be inferred."
        ),
        (
            "SENTIMENT_ALIGNMENT: inspect forum discussion comments posted before proposal end only, and Judge whether aggregated forum sentiment supports or opposes the "
            "likely vote outcome, state whether the recommendation mirrors forum opinion, and "
            "explicitly reference that alignment when finalizing both `ai_final_conclusion` and "
            "`ai_final_reason`."
            "report total comments and counts of positive/negative/neutral sentiments, and "
            "summarize the first two comments briefly."
        ),               
          (
            "DECISION_STANCE: conclude whether the proposal aims 'To change' or keep 'Not change, Status quo', "
            "and populate `decision_stance` with that exact phrase."
        ),
        
        (
            "INCORPORATE: weave lessons from similar proposals and forum sentiment counts into "
            "ai_final_reason alongside market and timeline analytics."
        ),
        "Objective: Choose exactly one option from the proposal's `choices`.",
        "Fill every field of ProposalDecision. Do NOT use ex-post tally.",
        "ONCHAIN GOAL (optional if needed): total holders & top-100 concentration.",
        "ANCHOR: Use END timestamp as the event window for market/TVL.",
        "Do NOT include token price impact (ex-post) as a pro/for argument.",
        "If provided later, `ex_post_market_response` (or ex_post_* fields) are for reporting only and MUST NOT affect the recommendation.",
        "Never mention any post-event/post-vote market metrics in `ai_final_reason`. "
        "If such metrics exist, leave `ai_final_reason` unchanged and put them ONLY in "
        "`ex_post_market_response` (reporting field).",
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
        max_iter=100,
        verbose_agent=True,
        description="Governance vote recommendation for a Snapshot proposal (ex-post blind).",
        instructions=(
            "Return a ProposalDecision object. Use the provided CONTEXT (timeline metrics, adjacent analytics). "
            "Use available MCP tools to gather forum discussion, classify each pre-deadline comment as for/against/unclear with supporting rationale, "
            "and run sentiment analysis to report total comments plus positive/negative/neutral counts (summarize the first two comments). "
            "Find similar historical proposals, summarize their intended changes in 1–2 sentences each, record post-vote token price/TVL reactions, and "
            "treat declines as unsuccessful lessons to inform the current decision. "
            "Choose exactly one option from `choices` and set both label and index. "
            "Include the full `choices` in available_choices. "
            "Set event_start_utc and event_end_utc (copy end into event_time_utc). "
            "Set address_of_governance_token to the authoritative token address. "
            "State explicitly whether the proposal outcome recommends 'To change' or 'Not change, Status quo' and weave historical and sentiment insights into ai_final_reason."
            "Weigh forum sentiment against the recommended option, noting if sentiment suggests the "
            "vote does or does not reflect community views, and explicitly state in `ai_final_reason` "
            "whether the recommendation mirrors aggregated sentiment. "
            "Synthesize vote counts, analytics, and sentiment to justify a recommendation even if it "
            "diverges from the leading tally; assume your guidance may alter the outcome and focus on "
            "the organization’s durable benefit."
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
    
    def _stance_from_recommended(timeline: Dict[str, Any]) -> Optional[str]:
        idx = timeline.get("recommended_index") if isinstance(timeline, dict) else None
        if idx is None:
            return None
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            return None
        if idx_int == 0:
            return "To change"
        if idx_int == 1:
            return "Not change, Status quo"
        return None

    def _format_market_response(entry: Dict[str, Any]) -> str:
        price = entry.get("price_impact_pct")
        tvl = entry.get("tvl_impact_pct")
        parts: List[str] = []
        if isinstance(price, (int, float)):
            parts.append(f"token {price:+.2f}%")
        if isinstance(tvl, (int, float)):
            parts.append(f"TVL {tvl:+.2f}%")
        if parts:
            return "; ".join(parts)
        return "No market response captured"

    def _similarity_reason(entry: Dict[str, Any], current_author: Optional[str]) -> str:
        reasons: List[str] = []
        author = entry.get("author")
        similarity = entry.get("similarity")
        if similarity is None:
            similarity = entry.get("similarity_score")
        if author and current_author and str(author).lower() == str(current_author).lower():
            reasons.append("Same author as current proposal")
        if isinstance(similarity, (int, float)) and similarity > 0:
            qualifier = "High" if similarity >= 0.5 else "Moderate" if similarity >= 0.2 else "Related"
            reasons.append(f"{qualifier} text overlap (score {similarity:.2f})")
        if author and not reasons:
            reasons.append(f"Authored by {author}")
        if not reasons:
            reasons.append("Contextual linkage via adjacency analytics")
        return ". ".join(reasons)

    def _summarize_entry(entry: Dict[str, Any]) -> str:
        title = entry.get("title") or "Similar proposal"
        end_utc = entry.get("end_utc")
        timeline = entry.get("timeline_metrics") or {}
        unique_voters = timeline.get("unique_voters")
        recommended = timeline.get("recommended_index")
        sentences: List[str] = []
        sentences.append(
            f"{title} closed on {end_utc if end_utc else 'an unknown date'}."
        )
        detail_parts: List[str] = []
        if isinstance(unique_voters, (int, float)):
            detail_parts.append(f"{int(unique_voters)} unique voters")
        timeline_summary = timeline.get("summary")
        if timeline_summary:
            detail_parts.append(str(timeline_summary))
        if recommended is not None:
            detail_parts.append(f"recommended option index {recommended}")
        if detail_parts:
            sentences.append("Timeline analytics noted " + ", ".join(detail_parts) + ".")
        return " ".join(sentences)

    current_author = None
    if isinstance(context.timeline_metrics, dict):
        current_author = context.timeline_metrics.get("proposal_author")

    scored_similars: List[Tuple[float, SimilarProposalSummary]] = []
    for entry in context.adjacent_analytics:
        if not isinstance(entry, dict):
            continue
        similarity_val = entry.get("similarity")
        if similarity_val is None:
            similarity_val = entry.get("similarity_score")
        try:
            similarity_float = float(similarity_val)
        except (TypeError, ValueError):
            similarity_float = 0.0
        summary_text = _summarize_entry(entry)
        reason_text = _similarity_reason(entry, current_author)
        market_text = _format_market_response(entry)
        stance_text = _stance_from_recommended(entry.get("timeline_metrics") or {})
        similar_summary = SimilarProposalSummary(
            proposal_id=str(entry.get("id") or ""),
            title=entry.get("title"),
            end_utc=entry.get("end_utc"),
            summary=summary_text,
            similarity_reason=reason_text,
            market_response=market_text,
            change_stance=stance_text,
        )
        scored_similars.append((similarity_float, similar_summary))

    scored_similars.sort(key=lambda item: item[0], reverse=True)
    decision.similar_proposals = [item[1] for item in scored_similars[:2]]
    

    return DecisionAgentOutput(decision=decision, raw_result=raw_result, pretty_print=pretty)

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
