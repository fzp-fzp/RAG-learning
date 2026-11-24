from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, UPLOAD_DIR
from app.ingestion import ingest_documents
from app.loaders import SUPPORTED_SUFFIXES
from app.vector_store import VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest local documents into ChromaDB.")
    parser.add_argument("paths", nargs="*", type=Path, help="Documents to ingest.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    paths = args.paths or sorted(
        path for path in UPLOAD_DIR.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        print(f"No supported documents found in {UPLOAD_DIR}")
        return 1

    with VectorStore() as store:
        try:
            report = ingest_documents(
                paths,
                store=store,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
        except ValueError as exc:
            print(f"配置错误: {exc}", file=sys.stderr)
            return 2

        for source, chunks in report.ingested:
            print(f"OK   {source}: {chunks} chunks")
        for source, error in report.failures:
            print(f"FAIL {source}: {error}", file=sys.stderr)

        print(f"Total chunks in knowledge base: {report.total_chunks}")
        for source, count in report.source_counts.items():
            print(f"  {source}: {count}")
        return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
