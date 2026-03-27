"""
chains.py — Groq LLM + Langfuse setup for Ad Performance Agent.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

load_dotenv()

# Bridge LANGFUSE_BASE_URL -> LANGFUSE_HOST (required by some Langfuse versions)
langfuse_base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
os.environ.setdefault("LANGFUSE_HOST", langfuse_base_url)

# Langfuse callback handler — reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
# LANGFUSE_HOST automatically from environment (langfuse v3+ behaviour)
langfuse_handler = LangfuseCallbackHandler()

# Langfuse client — for prompt management
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=langfuse_base_url,
)

# Groq LLM — llama-3.3-70b-versatile
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)


def get_langfuse_prompt(name: str, fallback: str, **variables) -> str:
    """
    Retrieve a prompt from Langfuse by name and format it with variables.
    Falls back to the provided fallback string if Langfuse is unavailable.

    Args:
        name: Langfuse prompt name (e.g. "analyse-performance")
        fallback: Raw prompt template string used when Langfuse is unreachable
        **variables: Template variables to interpolate into the prompt

    Returns:
        Formatted prompt string ready to send to the LLM
    """
    try:
        prompt_obj = langfuse_client.get_prompt(name)
        return prompt_obj.compile(**variables)
    except Exception:
        # Graceful fallback — format the hardcoded template manually
        result = fallback
        for key, value in variables.items():
            result = result.replace("{{" + key + "}}", str(value))
        return result
