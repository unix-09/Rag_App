import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768  # all-mpnet-base-v2

_active_model = None
_model_type: str | None = None  # "gemini" or "fallback"


# ── Model initialisation ───────────────────────────────────────────────────────

def _probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        model.embed_query("probe")
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
        return model
    except Exception as e:
        logfire.warning(f"Gemini probe failed: {e}. Will use sentence-transformers fallback.")
        return None


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


def _is_quota_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(term in message for term in ("429", "rate", "quota", "resource_exhausted"))


def _switch_to_fallback() -> None:
    global _active_model, _model_type
    if _model_type != "fallback":
        logfire.warning("Gemini quota is unavailable; switching to local embeddings.")
        _active_model = _load_fallback()
        _model_type = "fallback"


def _embed_local(texts: list[str]) -> list[list[float]]:
    return _active_model.encode(texts, show_progress_bar=False).tolist()


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


def get_embedding_collection() -> str:
    """Keep vectors from different embedding models in dimension-compatible collections."""
    _init()
    if _model_type == "gemini":
        return settings.QDRANT_COLLECTION
    return f"{settings.QDRANT_COLLECTION}_local"


# ── Batch embedding with retry ─────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "gemini":
        # Exponential backoff: 1 s → 2 s → 4 s → 8 s (4 attempts total)
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))
                if is_rate_limit and attempt < 3:
                    wait = 2 ** attempt
                    logfire.warning(
                        f"Gemini rate limit hit — retrying in {wait}s "
                        f"(attempt {attempt + 1}/4)."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError("Gemini rate limit persisted after 4 attempts.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        try:
            return _active_model.embed_query(query)
        except Exception as error:
            if not _is_quota_error(error):
                raise
            _switch_to_fallback()
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    if _model_type == "fallback":
        return _embed_local(texts)

    all_embeddings: list[list[float]] = []
    try:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
                all_embeddings.extend(_embed_batch(batch))
    except Exception as error:
        if not _is_quota_error(error):
            raise
        _switch_to_fallback()
        # Re-embed the complete document with one model so every vector has the same dimension.
        return _embed_local(texts)
    return all_embeddings
