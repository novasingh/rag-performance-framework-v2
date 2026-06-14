"""
rq1_experiment/rag_system/pipeline.py
=======================================
End-to-end RAG pipeline: retrieve → generate.

Executes all queries against a single condition's FAISS index using
Gemma 3 27B, logging full outputs for downstream metric computation.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss

from ..config import RAW_OUTPUTS_DIR, CONDITION_META
from .embedder import DocumentEmbedder
from .generator import GemmaGenerator
from .retriever import Retriever

logger = logging.getLogger(__name__)


def _output_path(condition_id: str) -> Path:
    return RAW_OUTPUTS_DIR / f"{condition_id}_outputs.json"


class RAGPipeline:
    """
    Full RAG pipeline for a single experimental condition.

    Usage
    -----
    pipeline = RAGPipeline(condition_id="C1", index=index, docs=docs,
                           embedder=embedder, generator=generator)
    results  = pipeline.run(queries)
    """

    def __init__(
        self,
        condition_id: str,
        index: faiss.IndexFlatIP,
        docs: List[Dict],
        embedder: DocumentEmbedder,
        generator: GemmaGenerator,
        k: int = 5,
    ) -> None:
        self.condition_id = condition_id
        self.meta         = CONDITION_META[condition_id]
        self.retriever    = Retriever(index, docs, embedder, k=k)
        self.generator    = generator

    def run_single(self, query_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run one query through the full pipeline.

        Parameters
        ----------
        query_record : dict with keys:
            - query_id       : str
            - question       : str
            - query_type     : 'factual' | 'analytical' | 'comparative'
            - time_sensitive : bool
            - reference_answer : str (ground truth)
            - relevance_keywords : List[str] (for P@k judgements)

        Returns
        -------
        dict with all inputs + retrieval + generation outputs
        """
        question            = query_record["question"]
        relevance_keywords  = query_record.get("relevance_keywords", [])

        # ── Retrieval ────────────────────────────────────────────────────────
        t0         = time.perf_counter()
        retrieval  = self.retriever.retrieve(question, relevance_keywords)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        # ── Generation ──────────────────────────────────────────────────────
        gen_result = self.generator.generate(
            question=question,
            retrieved_docs=retrieval["retrieved_docs"],
        )

        # ── Assemble output record ──────────────────────────────────────────
        return {
            # ─ query info ─
            "query_id":          query_record.get("query_id", ""),
            "question":          question,
            "query_type":        query_record.get("query_type", ""),
            "time_sensitive":    query_record.get("time_sensitive", False),
            "domain":            self.meta["domain"],
            "volatility":        self.meta["volatility"],
            "freshness_window":  self.meta["freshness"],
            "source_config":     self.meta["source_config"],
            # ─ condition info ─
            "condition_id":      self.condition_id,
            # ─ reference ─
            "reference_answer":  query_record.get("reference_answer", ""),
            # ─ retrieval ─
            "retrieved_docs":    retrieval["retrieved_docs"],
            "similarities":      retrieval["similarities"],
            "relevance_labels":  retrieval["relevance_labels"],
            "source_diversity":  retrieval["source_diversity"],
            "avg_freshness":     retrieval["avg_freshness"],
            "retrieval_ms":      round(retrieval_ms, 2),
            # ─ generation ─
            "response":          gen_result["response"],
            "latency_ms":        gen_result["latency_ms"],
            "generation_error":  gen_result.get("error"),
            "total_ms":          round(retrieval_ms + gen_result["latency_ms"], 2),
        }

    def run(
        self,
        queries: List[Dict[str, Any]],
        resume: bool = True,
        save_every: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Run all queries for this condition.

        Parameters
        ----------
        queries    : list of query_record dicts
        resume     : if True, load existing outputs and skip already-processed queries
        save_every : checkpoint save frequency (every N queries)

        Returns
        -------
        list of output records
        """
        out_path = _output_path(self.condition_id)
        results: List[Dict[str, Any]] = []
        processed_ids: set = set()

        # ── Resume from checkpoint ───────────────────────────────────────────
        if resume and out_path.exists():
            try:
                results = json.loads(out_path.read_text(encoding="utf-8"))
                processed_ids = {r["query_id"] for r in results}
                logger.info(
                    "Resuming %s: %d/%d queries already done.",
                    self.condition_id, len(processed_ids), len(queries),
                )
            except Exception as exc:
                logger.warning("Could not load checkpoint for %s: %s", self.condition_id, exc)
                results = []
                processed_ids = set()

        pending = [q for q in queries if q.get("query_id") not in processed_ids]
        logger.info(
            "Running %s: %d queries remaining (of %d total).",
            self.condition_id, len(pending), len(queries),
        )

        for i, query_rec in enumerate(pending, 1):
            try:
                result = self.run_single(query_rec)
                results.append(result)
                logger.info(
                    "[%s] %d/%d | %s | latency=%.0fms | err=%s",
                    self.condition_id, i, len(pending),
                    query_rec.get("query_id", "?"),
                    result.get("total_ms", 0),
                    result.get("generation_error"),
                )
            except Exception as exc:
                logger.error(
                    "[%s] Query %s failed: %s",
                    self.condition_id, query_rec.get("query_id", "?"), exc,
                )

            # Checkpoint save
            if i % save_every == 0:
                out_path.write_text(
                    json.dumps(results, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.info("[%s] Checkpoint saved at %d queries. Sleeping 5 minutes to cool down APIs...", self.condition_id, i)
                time.sleep(300)

        # Final save
        out_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "[%s] Complete. %d outputs saved to %s",
            self.condition_id, len(results), out_path,
        )
        return results
