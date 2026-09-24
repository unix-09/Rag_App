import logfire
from openai import OpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


# Use the workspace integration created for this app. No inline config is sent.
PRIMARY_MODEL = f"@{settings.GROQ_SLUG}/openai/gpt-oss-120b"
portkey_client = OpenAI(
    api_key=settings.PORTKEY_API_KEY,
    base_url=PORTKEY_GATEWAY_URL,
    default_headers=createHeaders(
        api_key=settings.PORTKEY_API_KEY,
        metadata={"environment": "local", "service": "enterprise-rag"},
    ),
)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatOpenAI sends requests through the existing Portkey workspace integration.
    """
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=PRIMARY_MODEL,
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"
