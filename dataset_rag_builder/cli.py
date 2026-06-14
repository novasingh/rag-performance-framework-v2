from __future__ import annotations

import argparse
from pathlib import Path

from .config import COLLECTIONS
from .functions.dataset_builder import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured RQ1 dataset builder with RAW -> CLEANED -> FINAL stages.")
    parser.add_argument(
        "collection_name",
        nargs="?",
        choices=[cfg.name for cfg in COLLECTIONS],
        help="Optional single collection name (example: healthcare_academic).",
    )
    parser.add_argument("--output-dir", default="rag_dataset", help="Output directory.")
    parser.add_argument("--target-min", type=int, default=1000, help="Minimum docs per collection.")
    parser.add_argument("--target-max", type=int, default=2000, help="Maximum docs per collection.")
    parser.add_argument(
        "--news-provider",
        choices=["rss", "hybrid", "gdelt"],
        default="rss",
        help="News source mode: rss (Google/Bing RSS only), hybrid (RSS first + GDELT fallback), gdelt (GDELT first + RSS supplement).",
    )
    parser.add_argument("--max-rounds", type=int, default=6, help="Retry rounds to fill target counts.")
    parser.add_argument("--allow-partial", action="store_true", help="Allow completion below target_min.")
    parser.add_argument("--fresh", action="store_true", help="Delete old output directory first.")
    parser.add_argument(
        "--collection",
        default="all",
        choices=["all"] + [cfg.name for cfg in COLLECTIONS],
        help="Run all collections or a single collection only.",
    )
    parser.add_argument(
        "--on-existing",
        default="auto",
        choices=["auto", "ask", "skip", "recreate", "fill"],
        help="Action when collection output already exists.",
    )
    parser.add_argument(
        "--rq1-query-alignment",
        action="store_true",
        help="Use RQ1 query set (4 queries per domain; 12 total) as collection keywords.",
    )
    args = parser.parse_args()

    if args.target_min > args.target_max:
        raise ValueError("target_min must be <= target_max")

    effective_collection = args.collection_name or args.collection
    if args.collection_name and args.collection != "all" and args.collection != args.collection_name:
        raise ValueError("Provide either positional collection_name or --collection, not conflicting values.")

    build_dataset(
        output_dir=Path(args.output_dir),
        target_min=args.target_min,
        target_max=args.target_max,
        news_provider=args.news_provider,
        strict=not args.allow_partial,
        max_rounds=args.max_rounds,
        fresh=args.fresh,
        collection=effective_collection,
        on_existing=args.on_existing,
        rq1_query_alignment=args.rq1_query_alignment,
    )
