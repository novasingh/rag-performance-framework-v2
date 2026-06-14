from __future__ import annotations

import argparse
from pathlib import Path

from main import COLLECTIONS, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 data collection using build_rag_dataset pipeline.")
    parser.add_argument("--output-dir", default="rag_dataset", help="Base output folder.")
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
        help="Run all collections or one specific collection.",
    )
    parser.add_argument(
        "--on-existing",
        default="auto",
        choices=["auto", "ask", "skip", "recreate", "fill"],
        help="Action when selected collection output already exists.",
    )
    parser.add_argument(
        "--disable-rq1-query-alignment",
        action="store_true",
        help="Disable RQ1 query alignment mode (enabled by default).",
    )
    args = parser.parse_args()

    if args.target_min > args.target_max:
        raise ValueError("target_min must be <= target_max")

    build_dataset(
        output_dir=Path(args.output_dir),
        target_min=args.target_min,
        target_max=args.target_max,
        news_provider=args.news_provider,
        strict=not args.allow_partial,
        max_rounds=args.max_rounds,
        fresh=args.fresh,
        collection=args.collection,
        on_existing=args.on_existing,
        rq1_query_alignment=not args.disable_rq1_query_alignment,
    )


if __name__ == "__main__":
    main()
