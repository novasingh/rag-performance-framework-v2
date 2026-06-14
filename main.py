from __future__ import annotations

from dataset_rag_builder import COLLECTIONS, RQ1_CONDITIONS, build_dataset
from dataset_rag_builder.cli import main

__all__ = ["COLLECTIONS", "RQ1_CONDITIONS", "build_dataset", "main"]


if __name__ == "__main__":
    main()
