import argparse
import logging
import uuid
from pathlib import Path

from dotenv import load_dotenv

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QDrantStorage
from custom_types import RagUpsertResult


load_dotenv()

logger = logging.getLogger(__name__)


def ingest_pdf(
    pdf_path,
    source_id,
    qdrant_url,
    collection,
) -> RagUpsertResult:
    """Load, chunk, embed, and upsert a PDF into Qdrant."""
    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    source = source_id or str(path)
    logger.info("Loading and chunking PDF: %s", path)
    chunks = load_and_chunk_pdf(str(path))
    if not chunks:
        raise ValueError(f"No text chunks produced from {path}")

    logger.info("Embedding %s chunks", len(chunks))
    vectors = embed_texts(chunks)

    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{i}"))
        for i in range(len(chunks))
    ]
    payloads = [{"source": source, "text": chunks[i]} for i in range(len(chunks))]

    store_kwargs = {}
    if qdrant_url:
        store_kwargs["url"] = qdrant_url
    if collection:
        store_kwargs["collection"] = collection

    logger.info(
        "Upserting into Qdrant (url=%s, collection=%s)",
        store_kwargs.get("url", "http://localhost:6333"),
        store_kwargs.get("collection", "docs3"),
    )
    QDrantStorage(**store_kwargs).upsert(ids, vectors, payloads)

    return RagUpsertResult(ingested=len(chunks))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into Qdrant without Inngest."
    )
    parser.add_argument("pdf_path", help="Path to the PDF to ingest")
    parser.add_argument(
        "--source-id",
        help="Optional source identifier stored with each chunk (defaults to PDF path)",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant endpoint (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name (default: docs3)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    try:
        result = ingest_pdf(
            pdf_path=args.pdf_path,
            source_id=args.source_id,
            qdrant_url=args.qdrant_url,
            collection=args.collection,
        )
        logger.info("Ingest complete. %s chunks inserted.", result.ingested)
    except Exception as exc:
        logger.error("Ingest failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
