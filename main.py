"""
main.py — FastAPI application for Ad Performance Agent.

Endpoints:
  GET  /health          — liveness check
  POST /run-agent       — upload CSV, run LangGraph pipeline, return insights
  POST /run-agent-json  — same pipeline but accepts JSON body (for API testing)

Run with: uvicorn main:app --reload
"""

import io
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from tools import parse_csv
from agent import build_graph, make_initial_state


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ad Performance Agent",
    description=(
        "AI agent that analyses ad performance CSVs and generates "
        "creative improvement suggestions using LangGraph + Groq."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AdRecord(BaseModel):
    ad_id: str | None = None
    platform: str | None = None
    campaign: str | None = None
    spend: float | None = None
    impressions: float | None = None
    clicks: float | None = None
    conversions: float | None = None
    CTR: float | None = None
    CPC: float | None = None
    ROAS: float | None = None


class JsonAnalysisRequest(BaseModel):
    ads: list[AdRecord]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_csv_data(parsed: dict) -> None:
    """
    Raise HTTPException 400 if the parsed CSV is missing required columns.

    Args:
        parsed: Output dict from tools.parse_csv()

    Raises:
        HTTPException: 400 if 'spend' column is absent or no metric column present
    """
    required_at_least_one = ["impressions", "clicks", "conversions", "ROAS", "CTR"]
    found = parsed.get("columns_found", [])

    if "spend" not in found:
        raise HTTPException(
            status_code=400,
            detail="Invalid CSV: 'spend' column is required.",
        )

    if not any(col in found for col in required_at_least_one):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid CSV: at least one of "
                f"{required_at_least_one} must be present."
            ),
        )


def _run_pipeline(csv_data: dict) -> dict:
    """
    Execute the LangGraph pipeline and collect the final state.

    Args:
        csv_data: Output dict from tools.parse_csv()

    Returns:
        JSON-serialisable response dict

    Raises:
        HTTPException: 500 if the agent pipeline fails
    """
    try:
        graph = build_graph()
        initial_state = make_initial_state(csv_data)

        final_state = None
        for state in graph.stream(initial_state):
            # stream() yields {node_name: state} dicts — keep the last full state
            for node_name, node_state in state.items():
                final_state = node_state

        if final_state is None:
            raise ValueError("Pipeline produced no output state.")

        return {
            "status": "success",
            "summary": {
                "total_ads": csv_data.get("total_ads", 0),
                "total_spend": csv_data.get("total_spend", 0),
                "avg_roas": csv_data.get("avg_ROAS", 0),
                "avg_ctr": csv_data.get("avg_CTR", 0),
            },
            "top_performers": final_state.get("top_performers", []),
            "underperformers": final_state.get("underperformers", []),
            "platform_breakdown": final_state.get("platform_breakdown", {}),
            "insights": final_state.get("performance_insights", ""),
            "creative_suggestions": final_state.get("creative_suggestions", ""),
            "report": final_state.get("final_report", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {str(e)}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """
    Liveness check endpoint.

    Returns:
        Basic status dict confirming the service is running
    """
    return {
        "status": "running",
        "agent": "Ad Performance Agent",
        "version": "1.0.0",
    }


@app.post("/run-agent")
async def run_agent(file: UploadFile = File(...)):
    """
    Accept a CSV file upload, run the LangGraph analysis pipeline, and return insights.

    The CSV must contain at minimum a 'spend' column plus at least one of:
    impressions, clicks, conversions, ROAS, or CTR.

    Args:
        file: Multipart CSV file upload

    Returns:
        JSON with summary stats, top/underperformers, LLM insights, and full report

    Raises:
        HTTPException 400: Invalid or missing CSV columns
        HTTPException 500: Agent pipeline failure
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()

    try:
        csv_data = parse_csv(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    _validate_csv_data(csv_data)

    return _run_pipeline(csv_data)


@app.post("/run-agent-json")
def run_agent_json(request: JsonAnalysisRequest):
    """
    Accept ad data as a JSON list of ad records, run the same pipeline as /run-agent.

    Useful for API testing without needing to upload a CSV file.

    Args:
        request: JSON body with an 'ads' list of ad record objects

    Returns:
        Same JSON structure as /run-agent

    Raises:
        HTTPException 400: Validation failure
        HTTPException 500: Agent pipeline failure
    """
    if not request.ads:
        raise HTTPException(status_code=400, detail="'ads' list cannot be empty.")

    # Convert Pydantic models -> dicts, then build a CSV in memory for consistent parsing
    rows = [ad.model_dump(exclude_none=False) for ad in request.ads]
    df = pd.DataFrame(rows)

    csv_bytes = df.to_csv(index=False).encode("utf-8")

    try:
        csv_data = parse_csv(csv_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process ad data: {str(e)}")

    _validate_csv_data(csv_data)

    return _run_pipeline(csv_data)
