"""
rq1_experiment/rag_system/embedder.py
======================================
Sentence-transformer embedding wrapper with batch processing and caching.

Uses all-MiniLM-L6-v2 (384-dim) — already installed in .venv.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

import os
from ..config import EMBEDDING_DIM, EMBEDDING_MODEL_NAME, DO_API_KEY, GENERATOR_BACKEND

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """
    Wraps SentenceTransformer for consistent text embedding across the pipeline.
    Uses DigitalOcean API by default if GENERATOR_BACKEND is 'digitalocean',
    otherwise uses local HuggingFace model.
    Produces L2-normalised float32 vectors suitable for FAISS inner-product search.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name
        self.dim        = EMBEDDING_DIM
        self.use_do_api = os.getenv("FORCE_DO_EMBEDDINGS", "0") == "1"
        self.model      = None

        if self.use_do_api:
            logger.info("Using DigitalOcean API for embeddings (FORCE_DO_EMBEDDINGS=1).")
        else:
            logger.info("Using LOCAL SentenceTransformer for embeddings (fast, no rate limits).")
            try:
                # Suppress annoying huggingface hub/transformers warnings
                import warnings
                warnings.filterwarnings("ignore", category=UserWarning)
                
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                logger.info("Local Embedding model loaded. Dimension=%d", self.dim)
            except Exception as e:
                logger.warning("Failed to load local SentenceTransformer: %s", e)
                logger.info("Falling back to DigitalOcean Inference API for embeddings.")
                self.use_do_api = True

    def _do_embed_batch(self, texts: List[str]) -> np.ndarray:
        import requests
        import time
        from .generator import _do_rate_limited_sleep, tracker
        
        url = 'https://inference.do-ai.run/v1/embeddings'
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {DO_API_KEY}'}
        
        vectors = []
        # DO handles batching, but to be safe with payload sizes, we chunk
        chunk_size = 20
        for i in range(0, len(texts), chunk_size):
            _do_rate_limited_sleep()
            batch = texts[i:i+chunk_size]
            data = {'model': 'all-mini-lm-l6-v2', 'input': batch}
            
            attempt = 0
            max_retries = 3
            resp = None
            while attempt < max_retries:
                try:
                    resp = requests.post(url, headers=headers, json=data, timeout=60)
                    if resp.status_code == 429:
                        if attempt > 0:
                            logger.error("Rate limit still exhausted after delay in embedder. Exiting program.")
                            import sys
                            sys.exit(1)
                        logger.warning("DigitalOcean HTTP 429 Rate Limit Hit in Embedder! Sleeping for 35 minutes...")
                        time.sleep(35 * 60)
                        attempt += 1
                        continue
                    break
                except Exception as e:
                    if isinstance(e, SystemExit):
                        raise
                    attempt += 1
                    logger.warning("DO Embedding API error (attempt %d): %s — retrying", attempt, e)
                    if attempt >= max_retries:
                        raise
                    time.sleep(2.0 * attempt)
                    
            if resp and resp.status_code == 200:
                out = resp.json()
                usage = out.get("usage", {})
                tracker.add(usage.get("prompt_tokens", 0), 0, requests=1)
                
                for item in sorted(out['data'], key=lambda x: x['index']):
                    vectors.append(item['embedding'])
            else:
                logger.error("DO Embedding failed HTTP %s: %s", getattr(resp, 'status_code', 'N/A'), getattr(resp, 'text', 'N/A'))
                for _ in batch:
                    vectors.append([0.0] * self.dim)
            time.sleep(0.1) # Prevent hammering the API too hard on huge batches
            
        return np.array(vectors, dtype=np.float32)

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self.use_do_api:
            vectors = self._do_embed_batch(texts)
            if normalize:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0 # prevent div by zero
                vectors = vectors / norms
            return vectors
        else:
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
            return vectors.astype(np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Embed a single query string. Returns shape (1, dim)."""
        return self.embed_texts([query], normalize=normalize)

    def embed_documents_for_indexing(
        self,
        docs: List[dict],
        text_field: str = "text",
        title_field: str = "title",
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of document dicts.

        Concatenates title + text for richer representations.
        """
        texts: List[str] = []
        for doc in docs:
            title = (doc.get(title_field) or "").strip()
            body  = (doc.get(text_field)  or "").strip()
            # Use first 512 words of text to avoid excessively long inputs
            body_words = body.split()
            if len(body_words) > 512:
                body = " ".join(body_words[:512])
            combined = f"{title}. {body}" if title else body
            texts.append(combined)

        logger.info("Embedding %d documents …", len(texts))
        vectors = self.embed_texts(
            texts, batch_size=batch_size, show_progress=show_progress
        )
        logger.info("Embedding complete. Shape=%s", vectors.shape)
        return vectors
