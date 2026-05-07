"""
Offline build script for the financial knowledge graph.

Reads `chunk_metadata/chunk_metadata.json`, extracts entities per chunk, builds
the co-occurrence graph, and writes it to `chunk_metadata/kg.pkl`.

Usage:
    # From repo root, inside the container or with PYTHONPATH set:
    python scripts/build_kg.py
    python scripts/build_kg.py --no-spacy            # skip spaCy (faster)
    python scripts/build_kg.py --limit 1000          # smoke test on 1000 chunks
    python scripts/build_kg.py --output some/path.pkl

The script is idempotent — running it again overwrites the existing kg.pkl.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make `src/` importable when running from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.chunk_loader import ChunkMetadataLoader  # noqa: E402

from kg import (  # noqa: E402
    FinancialEntityExtractor,
    FinancialKnowledgeGraph,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_kg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the financial KG.")
    p.add_argument(
        "--metadata",
        default=str(ROOT / "chunk_metadata" / "chunk_metadata.json"),
        help="Path to chunk_metadata.json",
    )
    p.add_argument(
        "--output",
        default=str(ROOT / "chunk_metadata" / "kg.pkl"),
        help="Where to write the persisted graph",
    )
    p.add_argument(
        "--no-spacy",
        action="store_true",
        help="Skip spaCy NER (faster build, lower ORG coverage)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, only process the first N chunks (for smoke testing)",
    )
    p.add_argument(
        "--report-every",
        type=int,
        default=2000,
        help="Log progress every N chunks",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    metadata_path = Path(args.metadata)
    output_path = Path(args.output)
    if not metadata_path.exists():
        logger.error("Metadata file not found: %s", metadata_path)
        return 1

    logger.info("📂 Loading chunks from %s", metadata_path)
    # Skip FAISS — we only need text + metadata for KG construction.
    loader = ChunkMetadataLoader(str(metadata_path), faiss_path=str(metadata_path))
    loader.load(load_faiss=False)
    chunks = loader.chunks
    if args.limit:
        chunks = chunks[: args.limit]
    logger.info("📦 %d chunks to process", len(chunks))

    extractor = FinancialEntityExtractor(use_spacy=not args.no_spacy)
    logger.info("🔧 Extractor status: %s", extractor.status)

    kg = FinancialKnowledgeGraph()

    start = time.time()
    last_report = start
    total_entities = 0
    chunks_with_no_entities = 0

    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("content", "")
        metadata = chunk.get("metadata", {})

        try:
            entities = extractor.extract(text, metadata=metadata)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extract failed on chunk %s: %s", chunk_id, exc)
            entities = []

        if entities:
            kg.add_chunk(chunk_id, entities)
            total_entities += len(entities)
        else:
            chunks_with_no_entities += 1

        if idx % args.report_every == 0:
            now = time.time()
            rate = args.report_every / max(0.001, now - last_report)
            eta = (len(chunks) - idx) / max(0.001, rate)
            logger.info(
                "  ▶ %d / %d chunks | %.0f chunks/s | ETA %.0fs | nodes=%d edges=%d",
                idx,
                len(chunks),
                rate,
                eta,
                kg.graph.number_of_nodes(),
                kg.graph.number_of_edges(),
            )
            last_report = now

    elapsed = time.time() - start
    logger.info("✅ Build complete in %.1fs", elapsed)
    logger.info("   Total entities extracted: %d", total_entities)
    logger.info("   Chunks with zero entities: %d", chunks_with_no_entities)

    stats = kg.stats()
    logger.info("📊 Graph stats:")
    for k, v in stats.items():
        logger.info("   %s: %s", k, v)

    kg.save(output_path)

    # Quick sanity check: load back and verify
    loaded = FinancialKnowledgeGraph.load(output_path)
    if loaded is None:
        logger.error("Could not reload the KG we just wrote — investigate.")
        return 2
    assert loaded.graph.number_of_nodes() == kg.graph.number_of_nodes()
    logger.info("🔁 Reload sanity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
