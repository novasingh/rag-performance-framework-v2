"""
rq1_experiment/run_experiment.py
==================================
Main experiment runner for RQ1.

Steps:
  1. Test Gemma API connection
  2. Build FAISS indexes for all 12 conditions
  3. Generate query banks (all 3 domains) — or load existing
  4. Run all queries through the RAG pipeline for each condition
  5. Save raw outputs per condition

Usage:
    python -m rq1_experiment.run_experiment [--conditions C1 C2 ...] [--force-rebuild]
    python -m rq1_experiment.run_experiment --step queries      # only build query banks
    python -m rq1_experiment.run_experiment --step indexes      # only build FAISS indexes
    python -m rq1_experiment.run_experiment --step experiment   # only run RAG (needs indexes+queries)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).parent / "results" / "experiment.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("rq1_experiment.run_experiment")


def _ensure_results_dir() -> None:
    from rq1_experiment.config import (
        ANALYSIS_DIR, FAISS_INDEX_DIR, METRICS_DIR,
        PLOTS_DIR, QUERY_BANK_DIR, RAW_OUTPUTS_DIR,
    )
    for d in [RAW_OUTPUTS_DIR, METRICS_DIR, ANALYSIS_DIR, PLOTS_DIR, QUERY_BANK_DIR, FAISS_INDEX_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def step_test_api() -> bool:
    from rq1_experiment.rag_system.generator import GemmaGenerator
    gen = GemmaGenerator()
    ok  = gen.test_connection()
    if not ok:
        logger.error("Gemma API test FAILED. Check your GOOGLE_AI_API_KEY in .env")
    return ok


def step_build_indexes(conditions: Optional[List[str]] = None, force: bool = False) -> dict:
    from rq1_experiment.rag_system.embedder import DocumentEmbedder
    from rq1_experiment.rag_system.indexer import build_all_indexes
    embedder = DocumentEmbedder()
    return build_all_indexes(embedder=embedder, force_rebuild=force, conditions=conditions)


def step_build_query_banks(force: bool = False) -> dict:
    from rq1_experiment.rag_system.generator import GemmaGenerator
    from rq1_experiment.query_bank.query_generator import QueryBankGenerator
    gen = GemmaGenerator()
    qbg = QueryBankGenerator(gen)
    return qbg.generate_all(force_regen=force)


def step_run_experiment(
    indexes: dict,
    conditions: Optional[List[str]] = None,
) -> None:
    from rq1_experiment.config import ALL_CONDITIONS, CONDITION_META, DOMAINS
    from rq1_experiment.query_bank.query_generator import load_query_bank
    from rq1_experiment.rag_system.embedder import DocumentEmbedder
    from rq1_experiment.rag_system.generator import GemmaGenerator
    from rq1_experiment.rag_system.pipeline import RAGPipeline

    target = conditions or ALL_CONDITIONS
    embedder = DocumentEmbedder()
    generator = GemmaGenerator()

    # Load query banks per domain
    query_banks = {}
    for domain in DOMAINS:
        try:
            query_banks[domain] = load_query_bank(domain)
            logger.info("Loaded %d queries for domain=%s", len(query_banks[domain]), domain)
        except FileNotFoundError:
            logger.error("Missing query bank for domain=%s. Run --step queries first.", domain)

    for cid in target:
        if cid not in indexes:
            logger.warning("No index for %s — skipping", cid)
            continue

        meta   = CONDITION_META[cid]
        domain = meta["domain"]
        if domain not in query_banks:
            logger.warning("No queries for domain=%s (condition %s) — skipping", domain, cid)
            continue

        index, docs = indexes[cid]
        queries     = query_banks[domain]

        logger.info("=== Running pipeline for %s (%s, %s) — %d queries ===",
                    cid, domain, meta["freshness"], len(queries))

        pipeline = RAGPipeline(
            condition_id=cid,
            index=index,
            docs=docs,
            embedder=embedder,
            generator=generator,
        )
        pipeline.run(queries, resume=True, save_every=10)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ1 Experiment Runner — Gemma 3 27B + FAISS RAG pipeline"
    )
    parser.add_argument(
        "--step",
        choices=["all", "api-test", "indexes", "queries", "experiment"],
        default="all",
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        metavar="C",
        help="Specific condition IDs to run (e.g. C1 C2 C7). Default: all 12.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of FAISS indexes and query banks even if cached.",
    )
    args = parser.parse_args()

    _ensure_results_dir()
    logger.info("RQ1 Experiment Runner starting — step=%s", args.step)

    if args.step in ("all", "api-test"):
        logger.info("=== Step 1: Testing Gemma API ===")
        if not step_test_api():
            if args.step == "api-test":
                sys.exit(1)
            logger.warning("API test failed but continuing …")

    indexes = {}

    if args.step in ("all", "indexes"):
        logger.info("=== Step 2: Building FAISS Indexes ===")
        indexes = step_build_indexes(args.conditions, force=args.force_rebuild)

    if args.step in ("all", "queries"):
        logger.info("=== Step 3: Building Query Banks ===")
        step_build_query_banks(force=args.force_rebuild)

    if args.step in ("all", "experiment"):
        if not indexes:
            logger.info("Loading existing FAISS indexes …")
            indexes = step_build_indexes(args.conditions, force=False)

        logger.info("=== Step 4: Running RAG Experiment ===")
        step_run_experiment(indexes, args.conditions)

    logger.info("Run complete.")


if __name__ == "__main__":
    main()
