"""
RQ1 Experiment Package
======================
Full experimental pipeline for Research Question 1:

  "How do dataset freshness levels and source type configurations affect
   RAG system performance metrics (retrieval precision, response accuracy,
   hallucination rate) across different domain volatility levels?"

Modules
-------
query_bank/     — query construction, validation, reference answers
rag_system/     — embedder, FAISS indexer, retriever, Gemma 3 27B generator, pipeline
evaluation/     — all metrics (P@5, nDCG@5, BERTScore, DeBERTa NLI, ROUGE-L, METEOR …)
analysis/       — ANOVA, regression, Random Forest, cross-validation, visualizer
results/        — generated runtime outputs (raw outputs, metrics, plots)
"""

__version__ = "1.0.0"
