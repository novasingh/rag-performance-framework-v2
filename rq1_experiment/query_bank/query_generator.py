"""
rq1_experiment/query_bank/query_generator.py
=============================================
Generates the evaluation query bank.

Supports two modes:
1. HUMAN-WRITTEN queries loaded from a CSV file
2. LLM-GENERATED queries using llama-4-maverick (DigitalOcean)

Human-written query CSV format (place at query_bank/queries/human_queries.csv):
    domain,query_type,question,reference_answer,time_sensitive
    technology,factual,What is...?,The answer...,true
    healthcare,analytical,Compare...?,Synthesis...,false
    history,comparative,How does...?,Comparison...,false

When human queries are available, they take priority.
Any shortfall is filled by LLM-generated queries.

Strategy (LLM mode):
- 200 queries per domain
- ~40% factual, ~35% analytical, ~25% comparative
- ~20% time-sensitive
- Queries seeded from actual documents in dataset

LLM backend: llama-4-maverick via DigitalOcean Inference API (42 RPM).
"""
from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import (
    CONDITION_META,
    CONDITIONS_DIR,
    DOMAINS,
    QUERIES_PER_DOMAIN,
    QUERY_BANK_DIR,
    QUERY_TYPE_DISTRIBUTION,
    TIME_SENSITIVE_RATIO,
    GENERATOR_BACKEND,
)
from ..rag_system.generator import DigitalOceanBackend

logger = logging.getLogger(__name__)

random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Human-written query loader
# ─────────────────────────────────────────────────────────────────────────────

HUMAN_QUERIES_PATH = QUERY_BANK_DIR / "queries" / "human_queries.csv"


def load_human_queries() -> Dict[str, List[Dict]]:
    """
    Load human-written queries from CSV file.
    Expected CSV columns: domain,query_type,question,reference_answer,time_sensitive
    Returns dict mapping domain -> list of query dicts.
    """
    if not HUMAN_QUERIES_PATH.exists():
        logger.info("No human-written query file found at %s", HUMAN_QUERIES_PATH)
        return {}

    result: Dict[str, List[Dict]] = {d: [] for d in DOMAINS}
    with open(HUMAN_QUERIES_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = row.get("domain", "").strip().lower()
            if domain not in result:
                logger.warning("Unknown domain '%s' in human queries - skipping", domain)
                continue
            qtype = row.get("query_type", "").strip()
            question = row.get("question", "").strip()
            reference_answer = row.get("reference_answer", "").strip()
            time_sensitive = row.get("time_sensitive", "").strip().lower() in ("true", "1", "yes")

            if not question or not reference_answer:
                continue

            result[domain].append({
                "query_id": f"{domain}_human_{len(result[domain]) + 1:04d}",
                "domain": domain,
                "query_type": qtype if qtype in ("factual", "analytical", "comparative") else "factual",
                "question": question,
                "reference_answer": reference_answer,
                "time_sensitive": time_sensitive,
                "source": "human_written",
            })

    for domain, queries in result.items():
        if queries:
            logger.info("Loaded %d human-written queries for %s", len(queries), domain)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# LLM-generated query support
# ─────────────────────────────────────────────────────────────────────────────

_FACTUAL_PROMPT = """\
You are a research assistant generating evaluation questions for a RAG (Retrieval-Augmented Generation) system benchmark.

Below is a document excerpt from the {domain} domain:

DOCUMENT TITLE: {title}
DOCUMENT SOURCE: {source_name} ({source_type})
DOCUMENT DATE: {publication_date}
TEXT:
{text_excerpt}

Generate ONE factual question that:
1. Has a single, specific, verifiable answer contained in the text above.
2. Cannot be answered from general knowledge alone (requires this specific document).
3. Is NOT trivially answered by copying a sentence verbatim.
4. Is time-sensitive IF the answer would differ if the document were from a different time period.

Respond in this EXACT JSON format (no extra text):
{{
  "question": "<the question>",
  "reference_answer": "<concise factual answer from the document>",
  "time_sensitive": <true or false>,
  "relevance_keywords": ["<3-5 key terms from the document that a retrieval system should match>"],
  "parametric_knowledge_risk": "<low|medium|high - whether LLM could answer without this document>"
}}"""

_ANALYTICAL_PROMPT = """\
You are a research assistant generating evaluation questions for a RAG system benchmark.

Below are TWO document excerpts from the {domain} domain:

DOCUMENT 1 - TITLE: {title1} | SOURCE: {source_name1} | DATE: {date1}
{text1}

DOCUMENT 2 - TITLE: {title2} | SOURCE: {source_name2} | DATE: {date2}
{text2}

Generate ONE analytical question that:
1. Requires integrating information from BOTH documents to produce a coherent answer.
2. Asks for synthesis, summary, or evaluation - not a single fact.
3. Cannot be answered from general knowledge alone.
4. Is time-sensitive IF the answer would differ if both documents were from an earlier period.

Respond in this EXACT JSON format (no extra text):
{{
  "question": "<the analytical question>",
  "reference_answer": "<2-4 sentence answer synthesizing both documents>",
  "time_sensitive": <true or false>,
  "relevance_keywords": ["<4-6 key terms a retrieval system should match>"],
  "parametric_knowledge_risk": "<low|medium|high>"
}}"""

_COMPARATIVE_PROMPT = """\
You are a research assistant generating evaluation questions for a RAG system benchmark.

Below are TWO document excerpts from the {domain} domain representing different approaches or time periods:

DOCUMENT 1 - TITLE: {title1} | SOURCE: {source_name1} | DATE: {date1}
{text1}

DOCUMENT 2 - TITLE: {title2} | SOURCE: {source_name2} | DATE: {date2}
{text2}

Generate ONE comparative question that:
1. Asks to contrast or compare the two documents/approaches/time periods.
2. Requires information from both documents to answer fully.
3. Cannot be answered from general knowledge alone.

Respond in this EXACT JSON format (no extra text):
{{
  "question": "<the comparative question>",
  "reference_answer": "<2-3 sentence answer comparing both>",
  "time_sensitive": <true or false>,
  "relevance_keywords": ["<4-6 key terms>"],
  "parametric_knowledge_risk": "<low|medium|high>"
}}"""


def _load_domain_docs(domain: str) -> List[Dict]:
    seen_ids: set = set()
    docs: List[Dict] = []
    for cid, meta in CONDITION_META.items():
        if meta["domain"] != domain:
            continue
        doc_path = CONDITIONS_DIR / cid / "condition_documents.json"
        if not doc_path.exists():
            continue
        with open(doc_path, encoding="utf-8") as fh:
            cond_docs = json.load(fh)
        for d in cond_docs:
            uid = d.get("id") or d.get("url") or ""
            if uid and uid in seen_ids:
                continue
            if uid:
                seen_ids.add(uid)
            docs.append(d)
    logger.info("Loaded %d unique documents for domain=%s", len(docs), domain)
    return docs


def _sample_doc_excerpt(doc: Dict, max_chars: int = 1000) -> str:
    text = (doc.get("text") or "").strip()
    return text[:max_chars]


def _parse_json_response(text: str) -> Optional[Dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
    return None


def _tag_time_sensitivity(queries: List[Dict], domain: str) -> List[Dict]:
    target_ts = int(len(queries) * TIME_SENSITIVE_RATIO)
    current_ts = sum(1 for q in queries if q.get("time_sensitive") is True)
    if current_ts >= target_ts:
        return queries
    promote_pool = [
        i for i, q in enumerate(queries)
        if not q.get("time_sensitive")
        and q.get("parametric_knowledge_risk", "") in {"medium", "low"}
    ]
    random.shuffle(promote_pool)
    needed = target_ts - current_ts
    for idx in promote_pool[:needed]:
        queries[idx]["time_sensitive"] = True
    logger.info(
        "Time-sensitivity tagging for %s: %d/%d queries are time-sensitive.",
        domain,
        sum(1 for q in queries if q.get("time_sensitive")),
        len(queries),
    )
    return queries


# ─────────────────────────────────────────────────────────────────────────────
# QueryBankGenerator (LLM mode using llama-4-maverick via DigitalOcean)
# ─────────────────────────────────────────────────────────────────────────────

class QueryBankGenerator:
    """LLM-based query bank generator using llama-4-maverick via DigitalOcean."""

    def __init__(self, generator: DigitalOceanBackend) -> None:
        self.generator = generator

    def generate_raw(self, prompt: str) -> str:
        """Call the backend and return text response."""
        try:
            return self.generator.call(prompt)
        except Exception as exc:
            logger.error("Backend call failed: %s", exc)
            return ""

    def _generate_factual(self, doc: Dict, domain: str) -> Optional[Dict]:
        prompt = _FACTUAL_PROMPT.format(
            domain=domain,
            title=doc.get("title", ""),
            source_name=doc.get("source_name", ""),
            source_type=doc.get("source_type", ""),
            publication_date=doc.get("publication_date", ""),
            text_excerpt=_sample_doc_excerpt(doc),
        )
        text = self.generate_raw(prompt)
        parsed = _parse_json_response(text)
        if parsed:
            parsed["query_type"] = "factual"
            parsed["source_doc_id"] = doc.get("id", "")
            parsed["source_url"] = doc.get("url", "")
            parsed["domain"] = domain
        return parsed

    def _generate_analytical(self, doc1: Dict, doc2: Dict, domain: str) -> Optional[Dict]:
        prompt = _ANALYTICAL_PROMPT.format(
            domain=domain,
            title1=doc1.get("title", ""), source_name1=doc1.get("source_name", ""),
            date1=doc1.get("publication_date", ""),
            text1=_sample_doc_excerpt(doc1, 700),
            title2=doc2.get("title", ""), source_name2=doc2.get("source_name", ""),
            date2=doc2.get("publication_date", ""),
            text2=_sample_doc_excerpt(doc2, 700),
        )
        text = self.generate_raw(prompt)
        parsed = _parse_json_response(text)
        if parsed:
            parsed["query_type"] = "analytical"
            parsed["source_doc_ids"] = [doc1.get("id", ""), doc2.get("id", "")]
            parsed["source_urls"] = [doc1.get("url", ""), doc2.get("url", "")]
            parsed["domain"] = domain
        return parsed

    def _generate_comparative(self, doc1: Dict, doc2: Dict, domain: str) -> Optional[Dict]:
        prompt = _COMPARATIVE_PROMPT.format(
            domain=domain,
            title1=doc1.get("title", ""), source_name1=doc1.get("source_name", ""),
            date1=doc1.get("publication_date", ""),
            text1=_sample_doc_excerpt(doc1, 700),
            title2=doc2.get("title", ""), source_name2=doc2.get("source_name", ""),
            date2=doc2.get("publication_date", ""),
            text2=_sample_doc_excerpt(doc2, 700),
        )
        text = self.generate_raw(prompt)
        parsed = _parse_json_response(text)
        if parsed:
            parsed["query_type"] = "comparative"
            parsed["source_doc_ids"] = [doc1.get("id", ""), doc2.get("id", "")]
            parsed["source_urls"] = [doc1.get("url", ""), doc2.get("url", "")]
            parsed["domain"] = domain
        return parsed

    def generate_domain_queries(
        self,
        domain: str,
        n_queries: int = QUERIES_PER_DOMAIN,
        force_regen: bool = False,
    ) -> List[Dict]:
        out_path = QUERY_BANK_DIR / f"{domain}_queries.json"

        if not force_regen and out_path.exists():
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if len(existing) >= n_queries:
                logger.info("Query bank for %s already has %d queries", domain, len(existing))
                return existing
            queries = existing
        else:
            queries = []

        n_factual = int(n_queries * QUERY_TYPE_DISTRIBUTION["factual"])
        n_analytical = int(n_queries * QUERY_TYPE_DISTRIBUTION["analytical"])
        n_comparative = n_queries - n_factual - n_analytical

        already_factual = sum(1 for q in queries if q.get("query_type") == "factual")
        already_analytical = sum(1 for q in queries if q.get("query_type") == "analytical")
        already_comparative = sum(1 for q in queries if q.get("query_type") == "comparative")

        need_f = n_factual - already_factual
        need_a = n_analytical - already_analytical
        need_c = n_comparative - already_comparative

        docs = _load_domain_docs(domain)
        if len(docs) < 2:
            logger.error("Not enough documents for domain=%s", domain)
            return queries

        random.shuffle(docs)
        doc_iter = iter(docs * 10)

        def _next_doc() -> Dict:
            try:
                return next(doc_iter)
            except StopIteration:
                return random.choice(docs)

        logger.info("Generating for %s: F=%d, A=%d, C=%d remaining", domain, need_f, need_a, need_c)
        counter = {"f": 0, "a": 0, "c": 0, "fail": 0}

        while counter["f"] < need_f:
            doc = _next_doc()
            try:
                q = self._generate_factual(doc, domain)
                if q and q.get("question") and q.get("reference_answer"):
                    q["query_id"] = f"{domain}_factual_{len(queries) + 1:04d}"
                    queries.append(q)
                    counter["f"] += 1
                    if len(queries) % 10 == 0:
                        out_path.write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")
                else:
                    counter["fail"] += 1
            except Exception as exc:
                logger.warning("Factual gen failed for %s: %s", domain, exc)
                counter["fail"] += 1

        while counter["a"] < need_a:
            d1, d2 = _next_doc(), _next_doc()
            try:
                q = self._generate_analytical(d1, d2, domain)
                if q and q.get("question") and q.get("reference_answer"):
                    q["query_id"] = f"{domain}_analytical_{len(queries) + 1:04d}"
                    queries.append(q)
                    counter["a"] += 1
                    if len(queries) % 10 == 0:
                        out_path.write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")
                else:
                    counter["fail"] += 1
            except Exception as exc:
                logger.warning("Analytical gen failed for %s: %s", domain, exc)
                counter["fail"] += 1

        while counter["c"] < need_c:
            d1, d2 = _next_doc(), _next_doc()
            try:
                q = self._generate_comparative(d1, d2, domain)
                if q and q.get("question") and q.get("reference_answer"):
                    q["query_id"] = f"{domain}_comparative_{len(queries) + 1:04d}"
                    queries.append(q)
                    counter["c"] += 1
                    if len(queries) % 10 == 0:
                        out_path.write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")
                else:
                    counter["fail"] += 1
            except Exception as exc:
                logger.warning("Comparative gen failed for %s: %s", domain, exc)
                counter["fail"] += 1

        queries = _tag_time_sensitivity(queries, domain)
        out_path.write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Query bank for %s: %d queries saved (failures=%d)", domain, len(queries), counter["fail"])
        return queries

    def generate_all(
        self,
        n_queries: int = QUERIES_PER_DOMAIN,
        force_regen: bool = False,
    ) -> Dict[str, List[Dict]]:
        result: Dict[str, List[Dict]] = {}
        for domain in DOMAINS:
            logger.info("=== Generating query bank for domain: %s ===", domain)
            result[domain] = self.generate_domain_queries(domain, n_queries=n_queries, force_regen=force_regen)
        return result


def load_query_bank(domain: str) -> List[Dict]:
    """Load the saved query bank for a domain."""
    path = QUERY_BANK_DIR / f"{domain}_queries.json"
    if not path.exists():
        raise FileNotFoundError(f"No query bank found for domain={domain}. Run query generation first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_query_banks() -> Dict[str, List[Dict]]:
    return {domain: load_query_bank(domain) for domain in DOMAINS}