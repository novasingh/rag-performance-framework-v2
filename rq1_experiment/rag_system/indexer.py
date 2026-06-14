"""
rq1_experiment/rag_system/indexer.py
=====================================
Build and persist FAISS exact inner-product search indexes, one per condition.

Design choices (from Report.pdf Section 3.4.3):
- FAISS IndexFlatIP: exact inner-product search — deterministic, no stochastic variance.
- Indexes are saved to disk and reloaded on subsequent runs (expensive to rebuild).
- Each condition gets its own index containing only documents from that condition.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from ..config import ALL_CONDITIONS, CONDITIONS_DIR, FAISS_INDEX_DIR
from .embedder import DocumentEmbedder

logger = logging.getLogger(__name__)

# File name conventions inside FAISS_INDEX_DIR / <condition_id> /
_INDEX_FILE   = "index.faiss"
_DOCMAP_FILE  = "docmap.pkl"    # maps FAISS integer id -> original doc dict
_META_FILE    = "meta.json"


def _condition_docs_path(condition_id: str) -> Path:
    return CONDITIONS_DIR / condition_id / "condition_documents.json"


def _condition_index_dir(condition_id: str) -> Path:
    return FAISS_INDEX_DIR / condition_id


def load_condition_documents(condition_id: str) -> List[Dict]:
    """Load the condition_documents.json for a given condition."""
    path = _condition_docs_path(condition_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No condition documents found for {condition_id} at {path}. "
            "Run the data collection pipeline first."
        )
    with open(path, encoding="utf-8") as fh:
        docs = json.load(fh)
    logger.info("Loaded %d documents for %s", len(docs), condition_id)
    return docs


def build_condition_index(
    condition_id: str,
    embedder: DocumentEmbedder,
    force_rebuild: bool = False,
) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """
    Build (or load from cache) a FAISS index for *condition_id*.

    Parameters
    ----------
    condition_id  : one of C1–C12
    embedder      : DocumentEmbedder instance (shared across conditions)
    force_rebuild : if True, ignores cached index and rebuilds from scratch

    Returns
    -------
    (faiss_index, docs_list)
    """
    idx_dir   = _condition_index_dir(condition_id)
    idx_file  = idx_dir / _INDEX_FILE
    doc_file  = idx_dir / _DOCMAP_FILE
    meta_file = idx_dir / _META_FILE

    # ── Load from cache if available ──────────────────────────────────────────
    if not force_rebuild and idx_file.exists() and doc_file.exists():
        logger.info("Loading cached FAISS index for %s …", condition_id)
        index = faiss.read_index(str(idx_file))
        with open(doc_file, "rb") as fh:
            docs = pickle.load(fh)
        logger.info(
            "Loaded index for %s: %d vectors, %d docs", condition_id, index.ntotal, len(docs)
        )
        return index, docs

    # ── Build from scratch ────────────────────────────────────────────────────
    logger.info("Building FAISS index for %s …", condition_id)
    docs    = load_condition_documents(condition_id)
    vectors = embedder.embed_documents_for_indexing(docs, show_progress=True)

    # Exact inner-product index (cosine similarity on L2-normalised vectors)
    dim   = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)  # type: ignore[arg-type]

    # Persist
    idx_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_file))
    with open(doc_file, "wb") as fh:
        pickle.dump(docs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    meta_file.write_text(
        json.dumps({"condition_id": condition_id, "num_docs": len(docs), "dim": dim}, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "FAISS index built and saved for %s: %d vectors (dim=%d)", condition_id, index.ntotal, dim
    )
    return index, docs


def build_all_indexes(
    embedder: Optional[DocumentEmbedder] = None,
    force_rebuild: bool = False,
    conditions: Optional[List[str]] = None,
) -> Dict[str, Tuple[faiss.IndexFlatIP, List[Dict]]]:
    """
    Build FAISS indexes for all (or specified) conditions.

    Returns
    -------
    dict mapping condition_id -> (faiss_index, docs_list)
    """
    if embedder is None:
        embedder = DocumentEmbedder()

    target = conditions or ALL_CONDITIONS
    result: Dict[str, Tuple[faiss.IndexFlatIP, List[Dict]]] = {}

    for cid in target:
        try:
            result[cid] = build_condition_index(cid, embedder, force_rebuild=force_rebuild)
        except Exception as exc:
            logger.error("Failed to build index for %s: %s", cid, exc)

    logger.info("Indexes ready for %d conditions.", len(result))
    return result
