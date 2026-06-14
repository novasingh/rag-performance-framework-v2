from __future__ import annotations

import logging
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_MODEL_DIRNAME = "all-MiniLM-L6-v2"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _model_cache_root() -> Path:
    env_path = os.getenv("RAG_MODEL_CACHE_DIR", "").strip()
    if env_path:
        return Path(env_path)
    return _project_root() / ".model_cache"


def _local_model_dir() -> Path:
    return _model_cache_root() / LOCAL_MODEL_DIRNAME


def get_embedding_model() -> SentenceTransformer:
    local_dir = _local_model_dir()
    if local_dir.exists():
        return SentenceTransformer(str(local_dir), local_files_only=True)

    cache_root = _model_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        logging.info("Pre-downloading embedding model to local cache: %s", local_dir)
        model = SentenceTransformer(MODEL_ID, cache_folder=str(cache_root))
        local_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(local_dir))
        return SentenceTransformer(str(local_dir), local_files_only=True)
    except Exception as err:
        logging.warning("Could not prepare local embedding cache, falling back to hub model: %s", err)
        return SentenceTransformer(MODEL_ID)
