"""
tools.py — CSV parser and metric formatting tools for Ad Performance Agent.
"""

import io
import pandas as pd


EXPECTED_COLUMNS = [
    "ad_id", "platform", "campaign", "spend",
    "impressions", "clicks", "conversions", "CTR", "CPC", "ROAS",
]

NUMERIC_COLUMNS = ["spend", "impressions", "clicks", "conversions", "CTR", "CPC", "ROAS"]


def parse_csv(file_content: bytes) -> dict:
    """
    Parse a CSV file from raw bytes into a structured dict of ad metrics.

    Extracts standard ad performance columns, computes summary statistics,
    and identifies the top 3 and bottom 3 ads by ROAS.

    Args:
        file_content: Raw bytes of the uploaded CSV file

    Returns:
        dict with keys:
            - ads: list of row dicts
            - columns_found: list of column names present
            - total_spend: float
            - total_conversions: float
            - avg_CTR: float
            - avg_ROAS: float
            - total_ads: int
            - top_performers: list of top 3 ad dicts by ROAS
            - underperformers: list of bottom 3 ad dicts by ROAS
    """
    df = pd.read_csv(io.BytesIO(file_content))

    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Keep only known columns that are actually present
    columns_found = [c for c in EXPECTED_COLUMNS if c in df.columns]

    # Coerce numeric columns, replacing unparseable values with NaN then 0
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Build per-row list of dicts using only found columns
    ads = df[columns_found].to_dict(orient="records")

    # Summary statistics
    total_spend = float(df["spend"].sum()) if "spend" in df.columns else 0.0
    total_conversions = float(df["conversions"].sum()) if "conversions" in df.columns else 0.0
    avg_CTR = float(df["CTR"].mean()) if "CTR" in df.columns else 0.0
    avg_ROAS = float(df["ROAS"].mean()) if "ROAS" in df.columns else 0.0

    # Top / bottom performers by ROAS
    if "ROAS" in df.columns:
        sorted_df = df.sort_values("ROAS", ascending=False)
        top_performers = sorted_df.head(3)[columns_found].to_dict(orient="records")
        underperformers = sorted_df.tail(3)[columns_found].to_dict(orient="records")
    else:
        top_performers = []
        underperformers = []

    return {
        "ads": ads,
        "columns_found": columns_found,
        "total_spend": round(total_spend, 2),
        "total_conversions": round(total_conversions, 2),
        "avg_CTR": round(avg_CTR, 4),
        "avg_ROAS": round(avg_ROAS, 2),
        "total_ads": len(ads),
        "top_performers": top_performers,
        "underperformers": underperformers,
    }


def format_metrics_for_llm(parsed_data: dict) -> str:
    """
    Format parsed ad data as concise, readable text for LLM consumption.

    Caps output at 3 000 characters to stay within token budgets.

    Args:
        parsed_data: Output dict from parse_csv()

    Returns:
        Formatted string with summary stats, top performers, and underperformers
    """
    lines = []

    lines.append("=== AD PERFORMANCE SUMMARY ===")
    lines.append(f"Total Ads Analysed : {parsed_data['total_ads']}")
    lines.append(f"Total Spend        : ${parsed_data['total_spend']:,.2f}")
    lines.append(f"Total Conversions  : {parsed_data['total_conversions']}")
    lines.append(f"Average CTR        : {parsed_data['avg_CTR']:.2%}")
    lines.append(f"Average ROAS       : {parsed_data['avg_ROAS']:.2f}x")
    lines.append("")

    # Top performers table
    lines.append("=== TOP 3 PERFORMERS (by ROAS) ===")
    for ad in parsed_data.get("top_performers", []):
        lines.append(
            f"  [{ad.get('ad_id', 'N/A')}] {ad.get('platform', '')} | "
            f"{ad.get('campaign', '')} | "
            f"Spend: ${ad.get('spend', 0):,.0f} | "
            f"ROAS: {ad.get('ROAS', 0):.2f}x | "
            f"CTR: {ad.get('CTR', 0):.2%} | "
            f"CPC: ${ad.get('CPC', 0):.2f} | "
            f"Conv: {ad.get('conversions', 0)}"
        )
    lines.append("")

    # Underperformers table
    lines.append("=== BOTTOM 3 UNDERPERFORMERS (by ROAS) ===")
    for ad in parsed_data.get("underperformers", []):
        lines.append(
            f"  [{ad.get('ad_id', 'N/A')}] {ad.get('platform', '')} | "
            f"{ad.get('campaign', '')} | "
            f"Spend: ${ad.get('spend', 0):,.0f} | "
            f"ROAS: {ad.get('ROAS', 0):.2f}x | "
            f"CTR: {ad.get('CTR', 0):.2%} | "
            f"CPC: ${ad.get('CPC', 0):.2f} | "
            f"Conv: {ad.get('conversions', 0)}"
        )
    lines.append("")

    # All ads raw table
    lines.append("=== ALL ADS ===")
    for ad in parsed_data.get("ads", []):
        lines.append(
            f"  [{ad.get('ad_id', 'N/A')}] {ad.get('platform', '')} | "
            f"{ad.get('campaign', '')} | "
            f"Spend: ${ad.get('spend', 0):,.0f} | "
            f"Impressions: {ad.get('impressions', 0):,} | "
            f"Clicks: {ad.get('clicks', 0):,} | "
            f"Conv: {ad.get('conversions', 0)} | "
            f"CTR: {ad.get('CTR', 0):.2%} | "
            f"CPC: ${ad.get('CPC', 0):.2f} | "
            f"ROAS: {ad.get('ROAS', 0):.2f}x"
        )

    full_text = "\n".join(lines)

    # Cap at 3 000 chars for token efficiency
    if len(full_text) > 3000:
        full_text = full_text[:2997] + "..."

    return full_text
