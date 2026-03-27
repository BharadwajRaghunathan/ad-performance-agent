# Ad Performance Agent

An AI-powered marketing intelligence agent that ingests Meta/Google ad performance CSVs and returns structured insights with creative improvement suggestions via a FastAPI backend.

## Architecture

The agent runs as a four-node **LangGraph StateGraph** behind **FastAPI**.

1. **parse_data_node** — pandas parses the CSV, computes total spend, avg ROAS/CTR, ranks ads by ROAS, and splits by platform (Meta vs Google).
2. **analyse_performance_node** — retrieves past analyses from Chroma (Agentic RAG), then calls Groq `llama-3.3-70b-versatile` for a five-section structured analysis: overall health, top/bottom ad diagnosis, platform comparison, budget reallocation.
3. **generate_suggestions_node** — second LLM call producing five data-grounded creative improvements per underperformer in `AD / PROBLEM / SUGGESTION / EXPECTED IMPACT` format.
4. **store_memory_node** — assembles the final markdown report and persists it to Chroma for future retrieval.

All LLM calls are traced in **LangSmith** and logged in **Langfuse** via its callback handler.

## Agentic RAG

Every completed analysis is embedded with HuggingFace `all-MiniLM-L6-v2` and stored in Chroma's `ad_analyses` collection. Before each LLM call, `analyse_performance_node` queries `retrieve_similar()` for the top-3 semantically similar past campaigns and injects them as grounding context. This improves multi-step reasoning — instead of analysing each CSV in isolation, the agent identifies cross-campaign patterns, improving recall of platform-level trends and reducing hallucination by anchoring outputs in verified historical data.

## Knowledge Graph Integration

The agent's suggestions are grounded in a marketing domain model. Entities: Ad Platform, Campaign Type, Audience Segment, Creative Type. Relationships: `Platform SUPPORTS CampaignType → TARGETS AudienceSegment → RESPONDS_TO CreativeType`. These relationships improve relevance — Meta Retargeting audiences respond differently to creatives than Google Brand Awareness audiences. In production this would be implemented in **Neo4j** and queried at suggestion-generation time.

## Evaluation Strategy

| Metric | Type |
|---|---|
| Hallucination rate — all referenced ad IDs must exist in the CSV | Automated |
| Relevance — suggestions must cite actual underperforming ad IDs | Automated |
| ROUGE score — summary quality vs human-written baseline | Automated |
| Suggestion quality — rubric review by a marketing expert | Manual |
| Latency and token usage — per-node breakdown in LangSmith | Automated |

## Pattern Recognition + Improvement Loop

Chroma memory accumulates every analysis; future runs retrieve similar past campaigns and inject them as context. Langfuse versions both prompts (`analyse-performance`, `generate-suggestions`) enabling A/B testing between iterations. When LangSmith flags a degraded run, the prompt is revised in Langfuse and live on the next call — no redeployment needed.

## Challenges + Solutions

1. **CSV inconsistency** → pandas normalises column names and coerces numerics before any LLM call.
2. **LLM hallucination** → raw numbers injected directly into the prompt; the model never recalls metrics from internal memory.
3. **Token limits** → metrics text hard-capped at 3,000 characters in `format_metrics_for_llm()`.

## Tech Stack

| Layer | Tool |
|---|---|
| Agent Framework | LangGraph |
| LLM | Groq llama-3.3-70b-versatile |
| Backend | FastAPI |
| Vector Memory | Chroma + HuggingFace all-MiniLM-L6-v2 |
| Observability | LangSmith + Langfuse |
| Data Processing | pandas |

## Results + Screenshots

### FastAPI Response — POST /run-agent
![FastAPI Output](results%20screenshots/fastapi%20output.png)

### LangSmith — 4-Node Pipeline Trace
![LangSmith Nodes](results%20screenshots/langsmith%20nodes.png)

### LangSmith — Trace Overview
![LangSmith Traces](results%20screenshots/langsmith%20traces.png)

### Langfuse — 2 LLM Calls Logged
![Langfuse](results%20screenshots/langfuse.png)

## API Response Explained

When you call `POST /run-agent` with a CSV file, the agent returns a single JSON object with these fields:

| Field | What it contains |
|---|---|
| `status` | `"success"` if the pipeline completed without errors |
| `summary` | High-level stats: total ads analysed, total spend, avg ROAS, avg CTR |
| `top_performers` | Top 3 ads ranked by ROAS — full metrics for each |
| `underperformers` | Bottom 3 ads ranked by ROAS — these are the focus of creative suggestions |
| `platform_breakdown` | Meta vs Google split: count, total spend, total conversions, avg ROAS per platform |
| `insights` | **LLM Call 1 output** — 5-section structured analysis: overall health, top ad diagnosis, underperformer root cause, platform comparison, budget reallocation recommendation |
| `creative_suggestions` | **LLM Call 2 output** — 5 actionable fixes for underperformers, each with AD / PROBLEM / SUGGESTION / EXPECTED IMPACT |
| `report` | Full markdown report combining all of the above, also saved to Chroma for future RAG retrieval |

**Example from the 11-ad sample run:**
- Total spend: **$11,570** across 11 ads
- Best performer: **AD003** (Meta Retargeting) at **8.15x ROAS**, CTR 4.00%, CPC $0.43
- Worst performer: **AD005** (Meta Brand Awareness) at **0.85x ROAS** — actively losing money
- Meta slightly trails Google: **4.18x vs 4.19x avg ROAS**
- LLM correctly identified brand awareness ads as the budget drain and recommended shifting spend to retargeting and lookalike audience campaigns

## How to Run

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
uvicorn main:app --reload
```

Test with the bundled sample CSV:

```bash
curl -X POST http://localhost:8000/run-agent \
  -F "file=@sample_data/sample_ads.csv"
```

Interactive API docs at `http://localhost:8000/docs`.
