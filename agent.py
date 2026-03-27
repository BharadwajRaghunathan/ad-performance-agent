"""
agent.py — LangGraph StateGraph for Ad Performance Agent.

4-node pipeline:
  parse_data_node -> analyse_performance_node
  -> generate_suggestions_node -> store_memory_node
"""

import datetime
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from chains import llm, langfuse_handler, get_langfuse_prompt
from tools import format_metrics_for_llm
from memory import save_analysis, retrieve_similar


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AdAnalysisState(TypedDict):
    raw_csv_data: dict          # output of parse_csv()
    metrics_summary: str        # formatted text for LLM
    performance_insights: str   # Node 2 LLM output
    creative_suggestions: str   # Node 3 LLM output
    final_report: str           # combined markdown report
    top_performers: list        # top 3 ads by ROAS
    underperformers: list       # bottom 3 ads by ROAS
    platform_breakdown: dict    # Meta vs Google split
    status_log: list            # step-by-step tracking


# ---------------------------------------------------------------------------
# Prompt templates (fallbacks if Langfuse unreachable)
# ---------------------------------------------------------------------------

ANALYSE_PERFORMANCE_FALLBACK = """You are an expert performance marketing analyst.

Analyse this ad performance data:
{{metrics_summary}}

Provide a structured analysis covering:
1. OVERALL PERFORMANCE ASSESSMENT
   Is this campaign healthy? Key numbers to note.
2. TOP PERFORMING ADS
   Why are these working? What patterns do you see?
3. UNDERPERFORMING ADS
   Why are these failing? What is the root cause?
4. PLATFORM COMPARISON
   Meta vs Google — which is delivering better ROI?
5. BUDGET REALLOCATION RECOMMENDATION
   Specific spend shifts to improve overall ROAS.

Be specific. Reference actual ad IDs and numbers.
Never make up metrics not present in the data."""


GENERATE_SUGGESTIONS_FALLBACK = """You are an expert ad creative strategist.

Based on this performance analysis:
{{insights}}

Raw data summary:
{{metrics_summary}}

Generate 5 specific, actionable creative improvement suggestions for the underperforming ads.

For each suggestion use this format:
AD: [ad_id]
PROBLEM: [one line — what is failing]
SUGGESTION: [specific creative change to make]
EXPECTED IMPACT: [what metric should improve and by how much]

Ground every suggestion in the actual data.
Do not make generic recommendations."""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def parse_data_node(state: AdAnalysisState) -> AdAnalysisState:
    """
    Node 1 — Format raw CSV data and compute platform breakdown.

    Reads raw_csv_data from state, calls format_metrics_for_llm(), and
    splits ads by platform (Meta vs Google).
    """
    log = list(state.get("status_log", []))
    log.append("parse_data_node: started")

    raw = state["raw_csv_data"]

    # Format metrics text for LLM
    metrics_summary = format_metrics_for_llm(raw)

    # Platform breakdown
    platform_breakdown: dict = {}
    for ad in raw.get("ads", []):
        platform = str(ad.get("platform", "Unknown")).strip()
        if platform not in platform_breakdown:
            platform_breakdown[platform] = {
                "count": 0,
                "total_spend": 0.0,
                "total_conversions": 0.0,
                "roas_values": [],
            }
        platform_breakdown[platform]["count"] += 1
        platform_breakdown[platform]["total_spend"] += float(ad.get("spend", 0))
        platform_breakdown[platform]["total_conversions"] += float(ad.get("conversions", 0))
        roas = ad.get("ROAS")
        if roas is not None:
            platform_breakdown[platform]["roas_values"].append(float(roas))

    # Compute avg ROAS per platform
    for platform, data in platform_breakdown.items():
        roas_vals = data.pop("roas_values", [])
        data["avg_roas"] = round(sum(roas_vals) / len(roas_vals), 2) if roas_vals else 0.0
        data["total_spend"] = round(data["total_spend"], 2)
        data["total_conversions"] = round(data["total_conversions"], 2)

    log.append(f"parse_data_node: formatted {raw['total_ads']} ads, {len(platform_breakdown)} platforms")

    return {
        **state,
        "metrics_summary": metrics_summary,
        "top_performers": raw.get("top_performers", []),
        "underperformers": raw.get("underperformers", []),
        "platform_breakdown": platform_breakdown,
        "status_log": log,
    }


def analyse_performance_node(state: AdAnalysisState) -> AdAnalysisState:
    """
    Node 2 — LLM call 1: deep performance analysis with Agentic RAG.

    Before calling the LLM, retrieves semantically similar past analyses from
    Chroma and injects them as additional context. This is the Agentic RAG step —
    the agent pulls relevant historical campaigns to ground its current analysis.

    Pulls the "analyse-performance" prompt from Langfuse (with fallback),
    invokes the Groq LLM with the Langfuse callback, and stores the result.
    """
    log = list(state.get("status_log", []))
    log.append("analyse_performance_node: started")

    # --- Agentic RAG: retrieve similar past analyses from Chroma ---
    past_context = ""
    try:
        similar = retrieve_similar(query=state["metrics_summary"], k=3)
        if similar:
            past_snippets = []
            for i, item in enumerate(similar, 1):
                # Include only the first 400 chars of each past report to stay token-efficient
                snippet = item["document"][:400].strip()
                meta = item.get("metadata", {})
                past_snippets.append(
                    f"Past Analysis {i} (campaign: {meta.get('campaign', 'unknown')}):\n{snippet}"
                )
            past_context = (
                "\n\nRELEVANT PAST ANALYSES (retrieved from memory for context):\n"
                + "\n\n".join(past_snippets)
                + "\n\nUse these only as background context. Do not copy them verbatim."
            )
            log.append(f"analyse_performance_node: injected {len(similar)} past analyses from Chroma")
        else:
            log.append("analyse_performance_node: no past analyses in Chroma yet")
    except Exception as e:
        log.append(f"analyse_performance_node: RAG retrieval skipped ({e})")

    # Append past context to metrics summary before prompt formatting
    enriched_summary = state["metrics_summary"] + past_context

    prompt_text = get_langfuse_prompt(
        name="analyse-performance",
        fallback=ANALYSE_PERFORMANCE_FALLBACK,
        metrics_summary=enriched_summary,
    )

    response = llm.invoke(
        prompt_text,
        config={"callbacks": [langfuse_handler]},
    )

    insights = response.content if hasattr(response, "content") else str(response)
    log.append("analyse_performance_node: LLM call complete")

    return {
        **state,
        "performance_insights": insights,
        "status_log": log,
    }


def generate_suggestions_node(state: AdAnalysisState) -> AdAnalysisState:
    """
    Node 3 — LLM call 2: creative improvement suggestions.

    Pulls the "generate-suggestions" prompt from Langfuse (with fallback),
    passes insights + metrics summary, and stores creative suggestions.
    """
    log = list(state.get("status_log", []))
    log.append("generate_suggestions_node: started")

    prompt_text = get_langfuse_prompt(
        name="generate-suggestions",
        fallback=GENERATE_SUGGESTIONS_FALLBACK,
        insights=state["performance_insights"],
        metrics_summary=state["metrics_summary"],
    )

    response = llm.invoke(
        prompt_text,
        config={"callbacks": [langfuse_handler]},
    )

    suggestions = response.content if hasattr(response, "content") else str(response)
    log.append("generate_suggestions_node: LLM call complete")

    return {
        **state,
        "creative_suggestions": suggestions,
        "status_log": log,
    }


def store_memory_node(state: AdAnalysisState) -> AdAnalysisState:
    """
    Node 4 — Combine insights + suggestions into a final report and persist to Chroma.

    Assembles a structured markdown report and saves it to the 'ad_analyses'
    Chroma collection. Silently skips storage if Chroma is unavailable.
    """
    log = list(state.get("status_log", []))
    log.append("store_memory_node: started")

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    final_report = f"""# Ad Performance Analysis Report
Generated: {timestamp}

---

## Performance Metrics Summary

{state['metrics_summary']}

---

## Performance Insights

{state['performance_insights']}

---

## Creative Improvement Suggestions

{state['creative_suggestions']}

---

*Report generated by Ad Performance Agent using Groq llama-3.3-70b-versatile*
"""

    # Save to Chroma — no-op if unavailable
    raw = state.get("raw_csv_data", {})
    campaign_name = f"analysis_{timestamp}"
    save_analysis(campaign_name, final_report)

    log.append("store_memory_node: report saved")

    return {
        **state,
        "final_report": final_report,
        "status_log": log,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """
    Build and compile the LangGraph StateGraph for ad performance analysis.

    Returns:
        Compiled LangGraph runnable ready for .stream() or .invoke() calls
    """
    builder = StateGraph(AdAnalysisState)

    builder.add_node("parse_data_node", parse_data_node)
    builder.add_node("analyse_performance_node", analyse_performance_node)
    builder.add_node("generate_suggestions_node", generate_suggestions_node)
    builder.add_node("store_memory_node", store_memory_node)

    builder.add_edge(START, "parse_data_node")
    builder.add_edge("parse_data_node", "analyse_performance_node")
    builder.add_edge("analyse_performance_node", "generate_suggestions_node")
    builder.add_edge("generate_suggestions_node", "store_memory_node")
    builder.add_edge("store_memory_node", END)

    return builder.compile()


def make_initial_state(csv_data: dict) -> AdAnalysisState:
    """
    Construct the initial AdAnalysisState from parsed CSV data.

    Args:
        csv_data: Output dict from tools.parse_csv()

    Returns:
        Fully initialised AdAnalysisState with empty LLM output fields
    """
    return AdAnalysisState(
        raw_csv_data=csv_data,
        metrics_summary="",
        performance_insights="",
        creative_suggestions="",
        final_report="",
        top_performers=[],
        underperformers=[],
        platform_breakdown={},
        status_log=[],
    )
