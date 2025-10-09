"""Streamlit interface for the governance ProposalDecision pipeline."""

from __future__ import annotations

import json
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

from agentics.core.llm_connections import available_llms
from examples import agentics_proposal_decision as apd
from examples.decision_agent_runner import ActualVoteResult, DecisionAgentContext, run_decision_agent

FOCUS_OPTIONS: List[str] = [
    "Token distribution and concentration",
    "Treasury impact and budget use",
    "Delegate turnout and voter participation",
    "Protocol risk (security, liquidity, TVL)",
    "Market impact (token price, volatility, trading volume)",
    "Governance process quality (discussion sentiment, delegate alignment)",
    "Long-term sustainability vs. short-term incentives",
]


def _resolve_focus(option: Optional[str], custom: str) -> Optional[str]:
    custom_clean = custom.strip()
    if custom_clean:
        return custom_clean
    if option and option != "(none)":
        return option
    return None


def _gather_forum_sentiment(
    tools, discussion_url: Optional[str]
) -> Optional[Dict[str, int]]:
    if not discussion_url:
        return None
    res = apd._invoke_tool_try_names_and_params(  # noqa: SLF001
        tools,
        ["forums_fetch_discussion", "fetch_discussion"],
        [{"url": discussion_url, "max_pages": 5}],
    )
    if not isinstance(res, dict):
        return None
    sentiment_summary = res.get("sentiment_summary")
    posts = res.get("posts")
    if not isinstance(sentiment_summary, dict) or not isinstance(posts, list):
        return None

    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return {
        "Negative": _safe_int(sentiment_summary.get("Negative")),
        "Positive": _safe_int(sentiment_summary.get("Positive")),
        "Neutral": _safe_int(sentiment_summary.get("Neutral")),
        "total_comments": len(posts),
    }


def _execute_pipeline(
    snapshot_url: str,
    focus: Optional[str],
    persist: bool,
) -> Dict[str, Any]:
    load_dotenv()
    project_root = Path(__file__).resolve().parent.parent
    registry_csv = (
        project_root / "src" / "agentics" / "assets_registry" / "dao_registry.csv"
    )

    with ExitStack() as stack:
        all_tools = apd._load_mcp_tools(project_root, stack)  # noqa: SLF001
        agent_tools = apd._restrict_agent_tools(  # noqa: SLF001
            apd._blind_toolset(all_tools)  # noqa: SLF001
        )

        llm_provider = apd.get_llm_provider()
        semantic_refs = apd._bootstrap_semantic_references(  # noqa: SLF001
            all_tools, project_root
        )

        pid = apd._resolve_pid_from_url(snapshot_url)  # noqa: SLF001
        if not pid:
            raise ValueError("Could not resolve proposal id from the provided URL.")

        meta_js = apd._fetch_meta_by_id(pid)  # noqa: SLF001
        result_js = apd._fetch_result_by_id(pid)  # noqa: SLF001

        title = meta_js.get("title")
        body = meta_js.get("body")
        author = meta_js.get("author")
        choices = meta_js.get("choices") or []
        discussion_url = meta_js.get("discussion")

        forum_sentiment_summary = _gather_forum_sentiment(all_tools, discussion_url)

        start_unix = int(meta_js.get("start") or 0)
        end_unix = int(meta_js.get("end") or 0)
        start_iso = apd._iso_from_unix(start_unix)  # noqa: SLF001
        end_iso = apd._iso_from_unix(end_unix)  # noqa: SLF001

        votes = apd._get_votes_via_snapshot_mcp(all_tools, pid)  # noqa: SLF001
        votes_count = len(votes)
        timeline_metrics = (
            apd._analyze_timeline(all_tools, start_unix, end_unix, choices, votes)
            if votes
            else {}
        )

        token_address = apd._extract_token_address_from_meta(meta_js)  # noqa: SLF001

        ucid = None
        defillama_slug = None
        defillama_type = None
        space = apd._space_from_url(snapshot_url)  # noqa: SLF001
        reg = apd._registry_csv_lookup_space_only(registry_csv, space=space)  # noqa: SLF001
        if reg:
            ucid = apd._clean_ucid(reg.get("cmc_ucid"))  # noqa: SLF001
            defillama_slug = reg.get("defillama_slug")
            defillama_type = reg.get("defillama_type")
        project_hint = apd._pick_project_hint(space, title)  # noqa: SLF001

        market_pre_days = 3
        market_post_days = 3

        price_metrics = apd.compute_market_adjusted_price_impact(  # noqa: SLF001
            project_root / "cmc_historical_daily_2013_2025.parquet",
            project_root / "SNP_cryptoIndex.xlsx",
            ucid,
            end_iso,
            pre_days=market_pre_days,
            post_days=market_post_days,
        )
        token_price_impact = price_metrics.get("token_pct")
        market_index_impact = price_metrics.get("market_pct")
        market_adjusted_price_impact = price_metrics.get("abnormal_pct")
        tvl_impact = apd.compute_tvl_impact_from_defillama_tool(
            all_tools,
            slug=defillama_slug,
            entity_type=defillama_type,
            project_hint=project_hint,
            event_end_utc=end_iso or "",
            pre_days=market_pre_days,
            post_days=market_post_days,
        )

        all_in_space = apd._fetch_all_proposals_by_space(space) if space else []  # noqa: SLF001
        adjacent = apd._select_adjacent_proposals(  # noqa: SLF001
            all_in_space,
            start_unix,
            current_author=author,
            current_title=title,
            current_body=body,
        )
        adjacent_analytics = apd._adjacent_analytics(  # noqa: SLF001
            all_tools,
            adjacent,
            cmc_parquet=project_root / "cmc_historical_daily_2013_2025.parquet",
            ucid=ucid,
            slug=defillama_slug,
            entity_type=defillama_type,
            project_hint=project_hint,
            current_title=title,
            current_body=body,
            market_index_path=project_root / "SNP_cryptoIndex.xlsx",
            max_items=3,
        )

        similar_data: List[Dict[str, Any]] = []
        similar_raw = None
        if space and pid:
            similar_raw = apd._invoke_tool_try_names_and_params(  # noqa: SLF001
                all_tools,
                ["snapshot_find_similar_proposals", "find_similar_proposals"],
                [
                    {"proposal_id": pid, "space": space, "max_days": 90, "max_n": 5},
                    {"proposal_id": pid, "space": space},
                ],
            )
            similar_raw = apd._normalize_tool_result(similar_raw)  # noqa: SLF001

        timeline_cache = {
            entry.get("id"): entry.get("timeline_metrics")
            for entry in adjacent_analytics
            if isinstance(entry, dict)
            and entry.get("id")
            and isinstance(entry.get("timeline_metrics"), dict)
        }
        adjacent_metric_cache = {
            entry.get("id"): entry
            for entry in adjacent_analytics
            if isinstance(entry, dict) and entry.get("id")
        }
        meta_cache: Dict[str, Dict[str, Any]] = {}
        result_cache: Dict[str, Dict[str, Any]] = {}

        if isinstance(similar_raw, list):
            for raw in similar_raw:
                if not isinstance(raw, dict):
                    continue
                sid = raw.get("id") or raw.get("proposal_id")
                if not sid:
                    continue

                timeline_metrics_sim = timeline_cache.get(sid)
                if timeline_metrics_sim is None:
                    meta_sim = meta_cache.get(sid)
                    if meta_sim is None:
                        meta_sim = apd._fetch_meta_by_id(sid) if sid else {}  # noqa: SLF001
                        meta_cache[sid] = meta_sim
                    choices_sim = (meta_sim or {}).get("choices") or []
                    start_sim = int((meta_sim or {}).get("start") or 0)
                    end_sim = int((meta_sim or {}).get("end") or 0)
                    votes_sim = apd._get_votes_via_snapshot_mcp(all_tools, sid)  # noqa: SLF001
                    if votes_sim and start_sim and end_sim:
                        timeline_metrics_sim = apd._analyze_timeline(  # noqa: SLF001
                            all_tools,
                            start_sim,
                            end_sim,
                            choices_sim,
                            votes_sim,
                        )
                        if timeline_metrics_sim:
                            timeline_cache[sid] = timeline_metrics_sim

                change_stance = apd._recommended_index_to_stance(  # noqa: SLF001
                    (timeline_metrics_sim or {}).get("recommended_index")
                    if isinstance(timeline_metrics_sim, dict)
                    else None
                )

                vote_result = raw.get("vote_result")
                vote_summary: Optional[Dict[str, Any]] = None
                if isinstance(vote_result, dict) and vote_result.get("scores"):
                    vote_summary = apd._summarize_vote_outcome(  # noqa: SLF001
                        vote_result.get("choices") or [],
                        vote_result.get("scores"),
                        vote_result.get("scores_total"),
                    )
                if vote_summary is None:
                    result_sim = result_cache.get(sid)
                    if result_sim is None:
                        result_sim = apd._fetch_result_by_id(sid)  # noqa: SLF001
                        result_cache[sid] = result_sim
                    meta_sim = meta_cache.get(sid)
                    if meta_sim is None:
                        meta_sim = apd._fetch_meta_by_id(sid)  # noqa: SLF001
                        meta_cache[sid] = meta_sim
                    vote_summary = apd._summarize_vote_outcome(  # noqa: SLF001
                        (result_sim or {}).get("choices")
                        or (meta_sim or {}).get("choices")
                        or [],
                        (result_sim or {}).get("scores"),
                        (result_sim or {}).get("scores_total"),
                    )

                adj_metrics = adjacent_metric_cache.get(sid, {})
                cleaned = {
                    "proposal_id": sid,
                    "title": raw.get("title"),
                    "author": raw.get("author"),
                    "end_utc": raw.get("end_utc"),
                    "similarity_score": raw.get("similarity_score"),
                    "winning_option": (
                        vote_summary.get("winner_label") if vote_summary else None
                    ),
                    "winning_option_index": (
                        vote_summary.get("winner_index") if vote_summary else None
                    ),
                    "margin_abs": (
                        vote_summary.get("margin_abs") if vote_summary else None
                    ),
                    "margin_pct": (
                        vote_summary.get("margin_pct") if vote_summary else None
                    ),
                    "scores_total": (
                        vote_summary.get("scores_total") if vote_summary else None
                    ),
                    "change_stance": change_stance,
                    "price_impact_pct": adj_metrics.get("price_impact_pct"),
                    "price_impact_market_pct": adj_metrics.get(
                        "price_impact_market_pct"
                    ),
                    "price_impact_market_adjusted_pct": adj_metrics.get(
                        "price_impact_market_adjusted_pct"
                    ),
                }

                similar_data.append(
                    {
                        "proposal_id": sid,
                        "cleaned": cleaned,
                        "timeline_metrics": (
                            timeline_metrics_sim if timeline_metrics_sim else None
                        ),
                        "price_impact_pct": adj_metrics.get("price_impact_pct"),
                        "price_impact_market_pct": adj_metrics.get(
                            "price_impact_market_pct"
                        ),
                        "price_impact_market_adjusted_pct": adj_metrics.get(
                            "price_impact_market_adjusted_pct"
                        ),
                        "raw": apd._normalize_similar_raw(raw),  # noqa: SLF001
                    }
                )

        agent_context = DecisionAgentContext(
            snapshot_url=snapshot_url,
            choices=choices,
            discussion_url=discussion_url,
            event_start_utc=start_iso,
            event_end_utc=end_iso,
            votes_count=votes_count,
            timeline_metrics=timeline_metrics,
            token_price_impact_pct=token_price_impact,
            token_price_market_pct=market_index_impact,
            token_price_market_adjusted_pct=market_adjusted_price_impact,
            tvl_impact_pct=tvl_impact,
            adjacent_analytics=adjacent_analytics,
            focus=focus,
        )

        decision_output = run_decision_agent(
            tools=agent_tools,
            llm_provider=llm_provider,
            context=agent_context,
        )

        decision = decision_output.decision

        def _coerce_choice(
            label: Optional[str],
            idx: Optional[int],
            options: List[str],
        ) -> tuple[Optional[str], Optional[int]]:
            if not options:
                return (label, idx)
            for i, opt in enumerate(options):
                if (label or "") == (opt or ""):
                    return (opt, i)
            norm = (label or "").strip().lower()
            for i, opt in enumerate(options):
                if norm == (opt or "").strip().lower():
                    return (opt, i)
            if idx is not None and 0 <= idx < len(options):
                return (options[idx], idx)
            return (label, idx)

        label, idx = _coerce_choice(
            decision.selected_choice_label,
            decision.selected_choice_index,
            choices,
        )
        decision.selected_choice_label = label
        decision.selected_choice_index = idx
        decision.available_choices = choices or decision.available_choices
        decision.event_start_utc = start_iso
        decision.event_end_utc = end_iso
        decision.event_time_utc = end_iso
        if token_address and not decision.address_of_governance_token:
            decision.address_of_governance_token = token_address

        if token_price_impact is not None:
            decision.token_price_impact_pct = token_price_impact
        if market_index_impact is not None:
            decision.token_price_market_pct = market_index_impact
        if market_adjusted_price_impact is not None:
            decision.token_price_market_adjusted_pct = market_adjusted_price_impact
        if tvl_impact is not None:
            decision.tvl_impact_pct = tvl_impact

        if token_price_impact is not None or tvl_impact is not None:
            decision.ex_post_price_impact_pct = token_price_impact
            decision.ex_post_tvl_impact_pct = tvl_impact
            decision.ex_post_window = (
                f"{market_pre_days}d pre / {market_post_days}d post around event end"
            )
            parts: List[str] = []
            if token_price_impact is not None:
                parts.append(f"Token price {token_price_impact:+.2f}%")
            if market_index_impact is not None:
                parts.append(f"Market index {market_index_impact:+.2f}%")
            if market_adjusted_price_impact is not None:
                parts.append(
                    f"Token vs market {market_adjusted_price_impact:+.2f}%"
                )
            if tvl_impact is not None:
                parts.append(f"TVL {tvl_impact:+.2f}%")
            decision.ex_post_note = "; ".join(parts) if parts else None

        scores = (result_js or {}).get("scores") or []
        scores_total = (result_js or {}).get("scores_total")
        actual = ActualVoteResult()
        actual.scores = [float(x) for x in scores] if scores else None
        actual.scores_total = float(scores_total) if scores_total is not None else None
        vote_summary = apd._summarize_vote_outcome(choices, scores, scores_total)  # noqa: SLF001
        if vote_summary:
            actual.winner_index = vote_summary.get("winner_index")
            actual.winner_label = vote_summary.get("winner_label")
            if vote_summary.get("margin_abs") is not None:
                actual.margin_abs = float(vote_summary["margin_abs"])
            if vote_summary.get("margin_pct") is not None:
                actual.margin_pct = round(float(vote_summary["margin_pct"]), 6)
        decision.actual_vote_result = actual

        agentic_choice = decision.selected_choice_label
        actual_outcome = (
            decision.actual_vote_result.winner_label
            if decision.actual_vote_result
            else None
        )

        def _match(a: Optional[str], b: Optional[str]) -> Optional[str]:
            if not a or not b:
                return None
            return "same" if a.strip().lower() == b.strip().lower() else "different"

        match_result = _match(agentic_choice, actual_outcome)

        payload: Dict[str, Any] = {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_url": snapshot_url,
            "focus": focus,
            "votes_count": votes_count,
            "timeline_metrics_current": timeline_metrics,
            "adjacent_analytics": adjacent_analytics,
            "similar_proposals_data": similar_data,
            "semantic_references": semantic_refs,
            "decision": decision.model_dump(),
            "market_index_impact_pct": market_index_impact,
            "market_adjusted_price_impact_pct": market_adjusted_price_impact,
            "agentic_ai_choice": agentic_choice,
            "actual_outcome": actual_outcome,
            "match_result": match_result,
        }
        if forum_sentiment_summary is not None:
            payload["forum_sentiment_summary"] = forum_sentiment_summary

        saved_path: Optional[Path] = None
        if persist:
            logdir = project_root / "Decision_runs"
            logdir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            saved_path = logdir / f"decision_{ts}.json"
            saved_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return {
            "decision": decision,
            "decision_output": decision_output,
            "timeline_metrics": timeline_metrics,
            "adjacent_analytics": adjacent_analytics,
            "similar_proposals": similar_data,
            "semantic_references": semantic_refs,
            "forum_sentiment": forum_sentiment_summary,
            "token_price_impact": token_price_impact,
            "market_index_impact": market_index_impact,
            "market_adjusted_price_impact": market_adjusted_price_impact,
            "tvl_impact": tvl_impact,
            "payload": payload,
            "saved_path": saved_path,
        }


st.set_page_config(
    page_title="Agentics Governance Demo",
    page_icon="🗳️",
    layout="wide",
)

st.title("Governance ProposalDecision Demo")
st.write(
    "Run the full Agentics governance decision workflow against a Snapshot proposal. "
    "Provide a proposal URL and optional focus area; the app will mirror the CLI example "
    "by collecting MCP data, building the long-form prompt, and returning the structured `ProposalDecision`."
)

if not available_llms:
    st.error(
        "No LLM providers detected. Configure credentials in `.env` (for example `OPENAI_API_KEY`)."
    )

st.sidebar.header("LLM Providers")
if available_llms:
    st.sidebar.markdown(
        "Configured providers: "
        + ", ".join(f"`{name}`" for name in available_llms.keys())
    )
else:
    st.sidebar.warning("Agentics will raise an error until an LLM provider is available.")

with st.sidebar.expander("Focus Areas", expanded=False):
    st.markdown("\n".join(f"- {item}" for item in FOCUS_OPTIONS))

with st.form("proposal_form"):
    snapshot_url = st.text_input(
        "Snapshot proposal URL",
        placeholder="https://snapshot.org/#/your-dao.eth/proposal/<id>",
    )
    focus_option = st.selectbox(
        "Focus area (optional)",
        options=["(none)"] + FOCUS_OPTIONS,
    )
    custom_focus = st.text_input(
        "Custom focus override",
        placeholder="Describe a specific concern to emphasize",
    )
    persist_choice = st.checkbox(
        "Persist run output to `Decision_runs/` like the CLI example",
        value=False,
    )
    submitted = st.form_submit_button("Run governance pipeline", type="primary")

if submitted:
    if not snapshot_url.strip():
        st.warning("Please enter a Snapshot proposal URL before running the pipeline.")
    elif not available_llms:
        st.warning("LLM provider missing; configure credentials and restart the app.")
    else:
        focus_value = _resolve_focus(focus_option, custom_focus)
        with st.spinner("Running Agentics governance pipeline..."):
            try:
                result_bundle = _execute_pipeline(
                    snapshot_url.strip(),
                    focus_value,
                    persist_choice,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Pipeline failed: {exc}")
            else:
                decision = result_bundle["decision"]
                decision_dict = decision.model_dump()
                st.success(
                    "ProposalDecision completed. Review the structured output and supporting analytics below."
                )

                col_decision, col_meta = st.columns([2, 1])

                with col_decision:
                    st.subheader("ProposalDecision")
                    st.json(decision_dict, expanded=False)
                    pretty = result_bundle["decision_output"].pretty_print
                    if pretty:
                        st.caption("Pretty-printed Agentics output")
                        st.code(pretty)

                with col_meta:
                    st.subheader("Impacts")
                    st.metric(
                        "Token price impact (%)",
                        f"{result_bundle['token_price_impact']:+.2f}" if result_bundle["token_price_impact"] is not None else "N/A",
                    )
                    st.metric(
                        "Market index impact (%)",
                        f"{result_bundle['market_index_impact']:+.2f}" if result_bundle.get("market_index_impact") is not None else "N/A",
                    )
                    st.metric(
                        "Token vs market (%)",
                        f"{result_bundle['market_adjusted_price_impact']:+.2f}" if result_bundle.get("market_adjusted_price_impact") is not None else "N/A",
                    )
                    st.metric(
                        "TVL impact (%)",
                        f"{result_bundle['tvl_impact']:+.2f}" if result_bundle["tvl_impact"] is not None else "N/A",
                    )
                    if result_bundle.get("forum_sentiment"):
                        st.subheader("Forum sentiment counts")
                        st.json(result_bundle["forum_sentiment"], expanded=False)

                st.subheader("Timeline metrics")
                st.json(result_bundle["timeline_metrics"], expanded=False)

                st.subheader("Adjacent analytics")
                st.json(result_bundle["adjacent_analytics"], expanded=False)

                st.subheader("Similar proposals (raw)")
                st.json(result_bundle["similar_proposals"], expanded=False)

                st.subheader("Semantic references")
                st.json(result_bundle["semantic_references"], expanded=False)

                payload_json = json.dumps(
                    result_bundle["payload"], ensure_ascii=False, indent=2
                )
                st.download_button(
                    label="Download full payload (JSON)",
                    data=payload_json,
                    file_name="proposal_decision_payload.json",
                    mime="application/json",
                )

                if result_bundle.get("saved_path"):
                    st.info(f"Saved run to {result_bundle['saved_path']}")

st.markdown("---")
st.caption(
    "Launch with `streamlit run examples/streamlit_demo.py`. Ensure MCP servers required by "
    "the CLI demo are available; the app reuses the same toolchain and logic."
)
