import os
import sys
import uuid
import json
import logfire
from dotenv import load_dotenv

load_dotenv()
logfire_token = os.getenv("LOGFIRE_TOKEN")
logfire.configure(
    token=logfire_token,
    send_to_logfire=bool(logfire_token),
    service_name="enterprise-ingestion-service",
    console=None if logfire_token else False,
)

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_texts, get_embedding_dim, get_embedding_collection
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_text

# Local folder where parsed + chunked JSON metadata is saved (replaces GCS processed bucket)
PROCESSED_DATA_DIR = "processed_data"
INGESTED_DOCUMENTS = 0
INDEXED_POINTS = 0
INGESTION_FAILURES: list[tuple[str, str]] = []

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def process_file(file_path: str, filename: str, source_type: str):
    """Parse → chunk → save locally → embed → index in Qdrant."""
    global INGESTED_DOCUMENTS, INDEXED_POINTS
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                return

            # 2. Chunk text
            chunks = chunk_text(full_text)
            if not chunks:
                return

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data → {local_path}")

            # 4. Embed and index in Qdrant
            with logfire.span("Vectorizing & Indexing"):
                embeddings = embed_texts(chunks)
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                        },
                    )
                    for chunk, vector in zip(chunks, embeddings)
                ]

                active_collection = get_embedding_collection()
                if not qdrant_client.collection_exists(active_collection):
                    qdrant_client.create_collection(
                        collection_name=active_collection,
                        vectors_config=models.VectorParams(
                            size=get_embedding_dim(),
                            distance=models.Distance.COSINE,
                        ),
                    )

                qdrant_client.upsert(
                    collection_name=active_collection,
                    points=points,
                )
                INGESTED_DOCUMENTS += 1
                INDEXED_POINTS += len(points)
                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}.")

        except Exception as e:
            INGESTION_FAILURES.append((filename, type(e).__name__))
            logfire.error("Failed to process {filename} ({error_type}).", filename=filename, error_type=type(e).__name__)


def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # Resolve the active embedding model first, then use its matching collection.
        dim = get_embedding_dim()
        collection_name = get_embedding_collection()

        # Wipe only the active model's collection if explicitly requested.
        if wipe:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(collection_name):
                    qdrant_client.delete_collection(collection_name)
                    logfire.info(f"Collection '{collection_name}' deleted.")

        # Create a collection sized for the active embedding model if needed.
        if not qdrant_client.collection_exists(collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection '{collection_name}' "
                f"({dim}-dim, Cosine)."
            )

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(os.path.join(base_dir, subdir), source_type)


if __name__ == "__main__":
    # Usage:
    #   python -m app.ingestion.processor DATA --wipe
    #   python -m app.ingestion.processor DATA/true_data true
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    print(f"Ingestion summary: {INGESTED_DOCUMENTS} documents indexed; {INDEXED_POINTS} points; {len(INGESTION_FAILURES)} failures.")
    for filename, error_type in INGESTION_FAILURES:
        print(f"Failed document: {filename} ({error_type}).")
    logfire.info("Ingestion job completed.")
