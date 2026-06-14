from __future__ import annotations

import json
import logging
import os
import random
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from dotenv import load_dotenv

from ..config import COLLECTIONS, DOMAIN_VOLATILITY_DAYS, RQ1_CONDITIONS, RQ1_DOMAIN_QUERIES, CollectionConfig, ConditionConfig
from ..functions.processing import (
    add_final_fields,
    clean_data,
    compute_freshness,
    deduplicate_data,
    is_news_doc_domain_relevant,
    looks_domain_relevant,
    looks_like_access_interstitial,
    looks_like_technology_documentation,
)
from ..functions.sources import (
    collect_history_documentation_seeds,
    collect_technology_documentation_seeds,
    download_arxiv_pdfs,
    fetch_article_text_playwright,
    fetch_article_text_playwright_with_url,
    fetch_article_text_requests,
    fetch_article_text_requests_with_url,
    parse_arxiv,
    parse_bing_news_rss,
    parse_crossref,
    parse_gdelt,
    parse_google_news_rss,
    parse_pubmed,
    parse_wikipedia,
    scrape_web_search,
)
from ..helpers.text_helpers import clean_text, normalize_text, parse_date
from ..utils.io_utils import load_json_list, now_iso, save_csv, save_json, setup_logging

try:
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover
    sync_playwright = None


def ensure_stage_structure(base_dir: Path) -> None:
    for stage in ["raw", "cleaned", "final"]:
        (base_dir / stage).mkdir(parents=True, exist_ok=True)
        for cfg in COLLECTIONS:
            (base_dir / stage / cfg.name).mkdir(parents=True, exist_ok=True)


def _unique_phrases(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        phrase = (value or "").strip()
        if not phrase:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out


_SHORT_TAIL_HINTS_BY_SOURCE: Dict[str, List[str]] = {
    "technical": [
        "overview",
        "basics",
        "guide",
        "reference",
        "examples",
        "workflow",
        "architecture",
        "best practices",
    ],
    "academic": [
        "study",
        "review",
        "method",
        "dataset",
        "evidence",
        "evaluation",
        "results",
        "framework",
    ],
    "news": [
        "today",
        "latest",
        "update",
        "analysis",
        "report",
        "trends",
        "outlook",
        "policy",
    ],
    "default": [
        "overview",
        "guide",
        "analysis",
        "reference",
        "best practices",
    ],
}


_LONG_TAIL_HINTS_BY_SOURCE: Dict[str, List[str]] = {
    "technical": [
        "official documentation",
        "developer guide",
        "api reference",
        "sdk integration tutorial",
        "configuration best practices",
        "implementation walkthrough",
        "deployment guide",
        "troubleshooting guide",
        "performance tuning guide",
        "security hardening guide",
        "architecture design patterns",
        "production readiness checklist",
        "step by step tutorial",
        "cli command reference",
        "integration examples",
        "migration guide",
    ],
    "academic": [
        "systematic review and meta analysis",
        "randomized controlled trial methodology",
        "cohort study design and outcomes",
        "longitudinal analysis and results",
        "comparative study with baseline",
        "experimental setup and evaluation metrics",
        "evidence synthesis and implications",
        "peer reviewed methodology overview",
        "causal inference approach and validation",
        "benchmark dataset evaluation",
        "ablation study and error analysis",
        "replication protocol and findings",
        "statistical significance and confidence intervals",
        "protocol design and ethical considerations",
        "implementation details and reproducibility",
        "state of the art literature review",
    ],
    "news": [
        "breaking news update",
        "latest policy update",
        "industry impact analysis",
        "market outlook report",
        "expert commentary",
        "regional impact report",
        "timeline and background",
        "fact check and analysis",
        "stakeholder response analysis",
        "data driven summary",
        "weekly roundup report",
        "emerging trend analysis",
        "regulatory change update",
        "incident impact assessment",
        "public response analysis",
        "in depth explainer",
    ],
    "default": [
        "in depth analysis",
        "comprehensive guide",
        "case study and methodology",
        "best practice framework",
        "implementation and outcomes",
    ],
}


def _keyword_pool_by_source(base_keywords: List[str], domain: str, source_type: str) -> List[str]:
    base = _unique_phrases(list(base_keywords))
    if not base:
        return []

    source_kind = (source_type or "").strip().lower() or "default"
    short_hints = _SHORT_TAIL_HINTS_BY_SOURCE.get(source_kind, _SHORT_TAIL_HINTS_BY_SOURCE["default"])
    long_hints = _LONG_TAIL_HINTS_BY_SOURCE.get(source_kind, _LONG_TAIL_HINTS_BY_SOURCE["default"])

    phrases = list(base)

    # Short-tail variants (concise search intent).
    for kw in base:
        phrases.append(f"{domain} {kw}")
        for hint in short_hints:
            phrases.append(f"{kw} {hint}")
            phrases.append(f"{domain} {kw} {hint}")

    # Long-tail variants (specific, multi-term search intent).
    for kw in base:
        for hint in long_hints:
            phrases.append(f"{kw} {hint}")
            phrases.append(f"{domain} {kw} {hint}")

    for query in RQ1_DOMAIN_QUERIES.get(domain, []):
        phrases.append(query)
        for hint in short_hints[:4]:
            phrases.append(f"{query} {hint}")
        for hint in long_hints[:4]:
            phrases.append(f"{query} {hint}")

    return _unique_phrases(phrases)


def _ensure_min_keyword_pool(pool: List[str], seed_terms: List[str], source_kind: str, min_size: int) -> List[str]:
    out = _unique_phrases(pool)
    if len(out) >= min_size:
        return out

    fallback = _unique_phrases(seed_terms) or ["research"]
    fillers = [
        "overview",
        "guide",
        "analysis",
        "reference",
        "methods",
        "examples",
        "comparison",
        "best practices",
        "case study",
        "deep dive",
    ]

    seen = {v.lower() for v in out}
    idx = 1
    max_attempts = max(2000, min_size * 20)
    while len(out) < min_size and idx <= max_attempts:
        term = fallback[(idx - 1) % len(fallback)]
        filler = fillers[(idx - 1) % len(fillers)]
        candidate = f"{term} {source_kind} {filler} query {idx}"
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            out.append(candidate)
        idx += 1

    return out


def _keyword_anchors(base_keywords: List[str], domain: str, source_kind: str) -> List[str]:
    base = _unique_phrases(base_keywords)
    lead = base[0] if base else domain
    anchors = [lead, f"{domain} {source_kind}".strip()]

    if source_kind == "technical":
        anchors.append(f"{domain} {lead} official documentation api reference guide")
    elif source_kind == "academic":
        anchors.append(f"{domain} {lead} systematic review and meta analysis methodology")
    elif source_kind == "news":
        anchors.append(f"{domain} {lead} breaking news update and impact analysis")
    else:
        anchors.append(f"{domain} {lead} in depth analysis and best practices")

    return _unique_phrases(anchors)


def expand_keywords(base_keywords: List[str], domain: str, round_num: int, source_type: str = "") -> List[str]:
    extras = [
        f"{domain} research",
        f"{domain} latest developments",
        f"{domain} review article",
        f"{domain} case study",
        f"{domain} analysis",
        str(now_iso()[:4]),
    ]

    source_kind = (source_type or "").strip().lower() or "default"
    expanded_pool = _keyword_pool_by_source(base_keywords, domain, source_kind)

    target_min = 250 + (round_num * 15)
    if source_kind == "technical" and (domain or "").strip().lower() == "technology":
        target_min += 120
    if source_kind == "news" and (domain or "").strip().lower() == "history":
        target_min += 60
    expanded_pool = _ensure_min_keyword_pool(
        expanded_pool,
        seed_terms=list(base_keywords) + list(RQ1_DOMAIN_QUERIES.get(domain, [])) + [domain],
        source_kind=source_kind,
        min_size=target_min,
    )

    random.shuffle(expanded_pool)
    k = min(len(expanded_pool), target_min)
    chosen = expanded_pool[:k]

    anchors = _keyword_anchors(base_keywords, domain, source_kind)
    return _unique_phrases(chosen + extras + anchors)


def _should_enrich_short_doc(cfg: CollectionConfig, doc: Dict, min_len: int = 300) -> bool:
    text = (doc.get("text") or "").strip()
    url = (doc.get("url") or "").strip().lower()
    if len(text) >= min_len or not url:
        return False

    # Academic APIs already provide abstracts; scraping DOI targets is slow and often fails.
    if cfg.source_type == "academic":
        return False
    if url.startswith("https://doi.org/") or url.startswith("http://doi.org/"):
        return False
    return True


_KEYWORDS_BY_DOMAIN_SOURCE: Dict[tuple[str, str], List[str]] = {
    (cfg.domain, cfg.source_type): cfg.keywords for cfg in COLLECTIONS
}


def _min_text_len_for_source(source_type: str) -> int:
    return 120 if source_type in {"academic", "news"} else 200


def _quality_drop_count(stats: Dict[str, int]) -> int:
    return sum(v for k, v in stats.items() if k.startswith("dropped_"))


def _strict_min_required(cfg: CollectionConfig, target_min: int) -> int:
    # Collections can plateau because of anti-bot pages, duplicate URL/content drops,
    # and relevance filtering. Keep strict mode meaningful for every category while
    # avoiding hard-failure after substantial progress.
    _ = cfg
    relaxed_floor = max(250, int(target_min * 0.4))
    return min(target_min, relaxed_floor)


_NEWS_TRACKING_QUERY_KEYS = {
    "gclid",
    "fbclid",
    "msclkid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ocid",
    "cmpid",
    "cmp",
    "ref",
    "ref_src",
    "ref_url",
    "output",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
}


def _canonicalize_news_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""

    try:
        parsed = urlparse(raw)
        scheme = (parsed.scheme or "https").lower()
        netloc = (parsed.netloc or "").lower()
        if not netloc:
            return raw.split("#", 1)[0].lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]

        path = parsed.path or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]

        kept_pairs = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            low_key = (key or "").strip().lower()
            if (not low_key) or low_key.startswith("utm_") or low_key in _NEWS_TRACKING_QUERY_KEYS:
                continue
            kept_pairs.append((key, value))

        query = urlencode(kept_pairs, doseq=True)
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return raw.split("#", 1)[0].lower()


def _url_key_for_source_type(url: str, source_type: str) -> str:
    source_kind = (source_type or "").strip().lower()
    if source_kind == "news":
        return _canonicalize_news_url(url)
    return (url or "").strip().lower()


def _relevance_min_hits(source_type: str, domain: str) -> int:
    source_kind = (source_type or "").strip().lower()
    domain_kind = (domain or "").strip().lower()
    if domain_kind == "history" and source_kind in {"academic", "technical", "news"}:
        return 1
    return 2


def _quality_filter_docs(
    docs: List[Dict],
    cfg: Optional[CollectionConfig] = None,
) -> tuple[List[Dict], Dict[str, int]]:
    stats: Dict[str, int] = {
        "dropped_non_dict": 0,
        "dropped_empty_title_or_text": 0,
        "dropped_short_text": 0,
        "dropped_domain_relevance": 0,
        "dropped_interstitial_page": 0,
        "dropped_non_documentation": 0,
        "dropped_duplicate_url": 0,
        "dropped_duplicate_content": 0,
    }

    filtered: List[Dict] = []
    seen_urls: Set[str] = set()
    seen_content_keys: Set[str] = set()

    for doc in docs:
        if not isinstance(doc, dict):
            stats["dropped_non_dict"] += 1
            continue

        record = dict(doc)
        record["title"] = clean_text(record.get("title", ""))
        record["text"] = clean_text(record.get("text", ""))

        if not record["title"] or not record["text"]:
            stats["dropped_empty_title_or_text"] += 1
            continue

        if looks_like_access_interstitial(
            title=record.get("title", ""),
            text=record.get("text", ""),
            url=record.get("url", ""),
        ):
            stats["dropped_interstitial_page"] += 1
            continue

        source_type = (record.get("source_type") or "").strip().lower()
        domain = (record.get("domain") or "").strip().lower()

        if cfg is not None:
            source_type = cfg.source_type
            domain = cfg.domain

        min_text_len = _min_text_len_for_source(source_type) if source_type else 120
        if len(record["text"]) < min_text_len:
            stats["dropped_short_text"] += 1
            continue

        keywords: List[str] = []
        if cfg is not None:
            keywords = cfg.keywords
        elif domain and source_type:
            keywords = _KEYWORDS_BY_DOMAIN_SOURCE.get((domain, source_type), [])

        if domain and keywords:
            blob = f"{record['title']} {record['text']}"
            relevance_min_hits = _relevance_min_hits(source_type, domain)
            if source_type == "news":
                if not is_news_doc_domain_relevant(
                    title=record["title"],
                    text=record["text"],
                    domain=domain,
                    keywords=keywords,
                    min_hits=relevance_min_hits,
                ):
                    stats["dropped_domain_relevance"] += 1
                    continue
            else:
                if not looks_domain_relevant(blob, domain, keywords, min_hits=relevance_min_hits):
                    stats["dropped_domain_relevance"] += 1
                    continue

        if source_type == "technical" and domain == "technology":
            source_name_low = (record.get("source_name") or "").strip().lower()
            if source_name_low == "wikipedia":
                stats["dropped_non_documentation"] += 1
                continue
            if source_name_low in {"googlesearch", "bingsearch"}:
                if not looks_like_technology_documentation(
                    title=record.get("title", ""),
                    text=record.get("text", ""),
                    url=record.get("url", ""),
                ):
                    stats["dropped_non_documentation"] += 1
                    continue

        url_key = _url_key_for_source_type(record.get("url", ""), source_type)
        if url_key:
            if url_key in seen_urls:
                stats["dropped_duplicate_url"] += 1
                continue
            seen_urls.add(url_key)

        content_key = normalize_text(f"{domain} {source_type} {record['title']} {record['text'][:1200]}")
        if content_key in seen_content_keys:
            stats["dropped_duplicate_content"] += 1
            continue
        seen_content_keys.add(content_key)

        filtered.append(record)

    return filtered, stats


_NEWS_PROVIDER_MODES = {"rss", "hybrid", "gdelt"}
_RAW_CHECKPOINT_EVERY = 25


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        logging.warning("Invalid int for %s=%r; using default=%s", name, raw, default)
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


_NEWS_SCRAPE_WAIT_MIN_MS = max(500, _env_int("RAG_NEWS_SCRAPE_WAIT_MIN_MS", 4000))
_NEWS_SCRAPE_WAIT_MAX_MS = max(_NEWS_SCRAPE_WAIT_MIN_MS, _env_int("RAG_NEWS_SCRAPE_WAIT_MAX_MS", 15000))
_HISTORY_NEWS_SCRAPE_WAIT_MIN_MS = max(500, _env_int("RAG_HISTORY_NEWS_SCRAPE_WAIT_MIN_MS", 1000))
_HISTORY_NEWS_SCRAPE_WAIT_MAX_MS = max(
    _HISTORY_NEWS_SCRAPE_WAIT_MIN_MS,
    _env_int("RAG_HISTORY_NEWS_SCRAPE_WAIT_MAX_MS", 3500),
)
_HISTORY_NEWS_SNIPPET_FALLBACK_MIN_CHARS = max(
    120,
    _env_int("RAG_HISTORY_NEWS_SNIPPET_FALLBACK_MIN_CHARS", 180),
)
_HISTORY_NEWS_RSS_GDELT_SUPPLEMENT = _env_flag("RAG_HISTORY_NEWS_RSS_GDELT_SUPPLEMENT", True)
_HISTORY_NEWS_MIN_RSS_CANDIDATES = max(2, _env_int("RAG_HISTORY_NEWS_MIN_RSS_CANDIDATES", 3))
_HISTORY_NEWS_GDELT_SUPPLEMENT_SIZE = max(6, _env_int("RAG_HISTORY_NEWS_GDELT_SUPPLEMENT_SIZE", 30))
_HISTORY_NEWS_SPARSE_WEB_SUPPLEMENT = _env_flag("RAG_HISTORY_NEWS_SPARSE_WEB_SUPPLEMENT", True)
_HISTORY_NEWS_WEB_SUPPLEMENT_RESULTS = max(4, _env_int("RAG_HISTORY_NEWS_WEB_SUPPLEMENT_RESULTS", 18))
_NEWS_SCRAPE_PROGRESS_LOG_EVERY = max(1, _env_int("RAG_NEWS_SCRAPE_PROGRESS_LOG_EVERY", 5))
_NO_GROWTH_KEYWORDS_TECHNICAL = max(6, _env_int("RAG_NO_GROWTH_KEYWORDS_TECHNICAL", 12))
_NO_GROWTH_KEYWORDS_NEWS = max(6, _env_int("RAG_NO_GROWTH_KEYWORDS_NEWS", 10))
_NO_GROWTH_KEYWORDS_ACADEMIC = max(8, _env_int("RAG_NO_GROWTH_KEYWORDS_ACADEMIC", 20))
_NO_GROWTH_KEYWORDS_DEFAULT = max(12, _env_int("RAG_NO_GROWTH_KEYWORDS_DEFAULT", 40))
_NO_GROWTH_STALLED_ROUNDS = max(1, _env_int("RAG_NO_GROWTH_STALLED_ROUNDS", 2))
_ACADEMIC_WIKIPEDIA_COOLDOWN_SECONDS = max(30, _env_int("RAG_ACADEMIC_WIKIPEDIA_COOLDOWN_SECONDS", 180))
_ACADEMIC_WIKIPEDIA_MAX_429_STREAK = max(1, _env_int("RAG_ACADEMIC_WIKIPEDIA_MAX_429_STREAK", 2))
_ACADEMIC_ALLOW_WEB_SEARCH_FALLBACK = _env_flag("RAG_ACADEMIC_ALLOW_WEB_SEARCH_FALLBACK", False)
_HISTORY_TECHNICAL_ALLOW_BROWSER_SEARCH = _env_flag("RAG_HISTORY_TECHNICAL_ALLOW_BROWSER_SEARCH", False)
_HISTORY_TECHNICAL_ALLOW_WIKIPEDIA_FALLBACK = _env_flag("RAG_HISTORY_TECHNICAL_ALLOW_WIKIPEDIA_FALLBACK", True)
_HISTORY_NEWS_PLAYWRIGHT_FALLBACK = _env_flag("RAG_HISTORY_NEWS_PLAYWRIGHT_FALLBACK", True)
_PUBMED_PAGES_PER_KEYWORD = max(1, _env_int("RAG_PUBMED_PAGES_PER_KEYWORD", 3))
_HEALTHCARE_ACADEMIC_C7_BACKFILL = _env_flag("RAG_HEALTHCARE_ACADEMIC_C7_BACKFILL", True)
_HEALTHCARE_ACADEMIC_C7_MIN_DOCS = max(0, _env_int("RAG_HEALTHCARE_ACADEMIC_C7_MIN_DOCS", 200))
_HEALTHCARE_ACADEMIC_PUBMED_MIN_AGE_DAYS = max(0, _env_int("RAG_HEALTHCARE_ACADEMIC_PUBMED_MIN_AGE_DAYS", 30))
_HEALTHCARE_ACADEMIC_PUBMED_MAX_AGE_DAYS = max(0, _env_int("RAG_HEALTHCARE_ACADEMIC_PUBMED_MAX_AGE_DAYS", 180))
_CONDITION_MIN_DOCS = max(1, _env_int("RAG_CONDITION_MIN_DOCS", 200))

_DOMAIN_NEWS_ANCHORS: Dict[str, str] = {
    "healthcare": "healthcare medical clinical hospital patient public health treatment",
    "technology": "technology software cloud cybersecurity ai digital platform",
    "history": "history historical archaeology heritage archive museum unesco artifact restoration",
}

_DOMAIN_NEWS_NEGATIVES: Dict[str, str] = {
    "healthcare": "-bitcoin -crypto -semiconductor -chip -nasdaq -stock",
    "history": "-live -score -match -ipl -cricket -football -soccer -nba -nfl -boxoffice -celebrity",
}


def _normalize_news_provider(news_provider: str) -> str:
    provider = (news_provider or "").strip().lower()
    if provider in _NEWS_PROVIDER_MODES:
        return provider
    logging.warning("Unknown news provider=%s; defaulting to rss", news_provider)
    return "rss"


def _news_scrape_wait_bounds(cfg: CollectionConfig) -> tuple[int, int]:
    if cfg.source_type == "news" and cfg.domain == "history":
        return _HISTORY_NEWS_SCRAPE_WAIT_MIN_MS, _HISTORY_NEWS_SCRAPE_WAIT_MAX_MS
    return _NEWS_SCRAPE_WAIT_MIN_MS, _NEWS_SCRAPE_WAIT_MAX_MS


def _academic_source_size(cfg: CollectionConfig, target_max: int) -> int:
    # Wikipedia-backed academic runs can trigger 429 bursts if each keyword requests too many pages.
    if cfg.primary_source == "wikipedia":
        return max(10, min(24, max(12, target_max // 85)))
    if cfg.primary_source == "pubmed":
        return max(18, min(45, max(22, target_max // 35)))
    return max(16, min(36, max(20, target_max // 45)))


def _technical_source_size(cfg: CollectionConfig, target_max: int) -> int:
    return max(12, min(36, max(16, target_max // 40)))


def _history_technical_web_target(source_size: int) -> int:
    return max(18, min(40, max(12, source_size + 4)))


def _history_technical_refill_target(docs_target: int) -> int:
    return max(14, min(48, docs_target + 8))


def _browser_for_collection(cfg: CollectionConfig, browser: Optional[Any]) -> Optional[Any]:
    if browser is None:
        return None
    if cfg.source_type == "technical" and cfg.domain == "history" and not _HISTORY_TECHNICAL_ALLOW_BROWSER_SEARCH:
        return None
    if cfg.source_type == "news" and cfg.domain == "history" and not _HISTORY_NEWS_PLAYWRIGHT_FALLBACK:
        return None
    return browser


def _dedupe_news_candidates(cfg: CollectionConfig, docs: List[Dict], limit: int) -> List[Dict]:
    deduped: List[Dict] = []
    seen_keys: Set[str] = set()
    cap = max(8, limit)

    for item in docs:
        if not isinstance(item, dict):
            continue

        title = clean_text(item.get("title", ""))
        text = clean_text(item.get("text", ""))
        source_name = (item.get("source_name") or "").strip().lower()

        url_key = _url_key_for_source_type(item.get("url", ""), "news")
        title_key = normalize_text(title)[:220]
        key = url_key or f"{source_name}::{title_key}"
        if (not key) or key in seen_keys:
            continue

        # Cheap pre-filter using RSS/GDELT snippet text to avoid scraping obviously off-topic links.
        if text:
            if not is_news_doc_domain_relevant(
                title=title,
                text=text,
                domain=cfg.domain,
                keywords=cfg.keywords,
                min_hits=1,
            ):
                continue

        row = dict(item)
        if title:
            row["title"] = title
        if text:
            row["text"] = text

        seen_keys.add(key)
        deduped.append(row)
        if len(deduped) >= cap:
            break

    return deduped


def _keyword_no_growth_limit(cfg: CollectionConfig, keyword_count: int) -> int:
    if cfg.source_type == "technical" and cfg.domain == "technology":
        return min(_NO_GROWTH_KEYWORDS_TECHNICAL, max(6, keyword_count // 80))
    if cfg.source_type == "technical" and cfg.domain == "history":
        # History technical queries can have poor yield due to anti-bot and strict filters, probe much deeper
        return min(max(_NO_GROWTH_KEYWORDS_TECHNICAL * 2, 14), max(14, keyword_count // 20))
    if cfg.source_type == "technical":
        return min(_NO_GROWTH_KEYWORDS_TECHNICAL, max(8, keyword_count // 25))
    if cfg.source_type == "news":
        if cfg.domain == "history":
            # History news queries can be sparse/duplicate-heavy; probe deeper before declaring plateau.
            return min(max(_NO_GROWTH_KEYWORDS_NEWS * 2, 14), max(14, keyword_count // 20))
        return min(_NO_GROWTH_KEYWORDS_NEWS, max(8, keyword_count // 14))
    if cfg.source_type == "academic":
        if cfg.primary_source == "wikipedia":
            return min(_NO_GROWTH_KEYWORDS_ACADEMIC, max(8, keyword_count // 25))
        return min(_NO_GROWTH_KEYWORDS_ACADEMIC, max(10, keyword_count // 12))
    return min(_NO_GROWTH_KEYWORDS_DEFAULT, max(12, keyword_count // 6))


def _news_query_for_domain(keyword: str, domain: str, include_negatives: bool = True) -> str:
    base = (keyword or "").strip()
    domain_key = (domain or "").strip().lower()
    anchors = _DOMAIN_NEWS_ANCHORS.get(domain_key, "")
    negatives = _DOMAIN_NEWS_NEGATIVES.get(domain_key, "") if include_negatives else ""
    parts = [p for p in [base, anchors, negatives] if p]
    return " ".join(parts)


def _collect_news_docs(
    session: requests.Session,
    cfg: CollectionConfig,
    keyword: str,
    source_size: int,
    news_provider: str,
) -> List[Dict]:
    docs: List[Dict] = []
    provider = _normalize_news_provider(news_provider)
    query = _news_query_for_domain(keyword, cfg.domain, include_negatives=True)
    gdelt_query = _news_query_for_domain(keyword, cfg.domain, include_negatives=False)

    def _append_google(limit: int, query_text: Optional[str] = None) -> None:
        effective_query = (query_text or query).strip() or query
        try:
            docs.extend(parse_google_news_rss(session, cfg, effective_query, size=limit))
        except Exception as err:
            logging.warning("Google News RSS failed for %s keyword=%s error=%s", cfg.name, effective_query, err)

    def _append_bing(limit: int, query_text: Optional[str] = None) -> None:
        effective_query = (query_text or query).strip() or query
        try:
            docs.extend(parse_bing_news_rss(session, cfg, effective_query, size=limit))
        except Exception as err:
            logging.warning("Bing News RSS failed for %s keyword=%s error=%s", cfg.name, effective_query, err)

    def _append_gdelt(limit: int, timespan: str = "3months") -> None:
        try:
            docs.extend(parse_gdelt(session, cfg, gdelt_query, timespan=timespan, max_records=limit))
        except Exception as err:
            logging.warning("GDELT failed for %s keyword=%s error=%s", cfg.name, query, err)

    def _history_gdelt_timespan(default: str = "3months") -> str:
        if cfg.source_type == "news" and cfg.domain == "history":
            return "12months"
        return default

    def _rss_queries() -> List[str]:
        if cfg.source_type == "news" and cfg.domain == "history":
            return _unique_phrases(
                [
                    query,
                    f"{keyword} history museum archive heritage archaeology",
                    f"{keyword} historical records unesco repatriation heritage site",
                    f"{keyword} archaeological discovery heritage preservation",
                    f"{keyword} museum exhibition artifact restoration",
                    f"{keyword} historical anniversary commemoration",
                    f"{keyword} heritage site restoration conservation",
                    f"{keyword} archive digitization museum records",
                    f"{keyword} archaeology excavation findings",
                    f"{keyword} repatriation restitution museum",
                ]
            )
        return [query]

    if provider == "gdelt":
        _append_gdelt(max(10, min(40, source_size)), timespan=_history_gdelt_timespan("3months"))
        rss_supplement = max(6, min(18, max(4, source_size // 4)))
        _append_google(rss_supplement)
        _append_bing(rss_supplement)
        return _dedupe_news_candidates(cfg, docs, limit=max(8, min(32, source_size)))

    # For RSS modes we split budget across Google and Bing to avoid oversized per-keyword batches.
    rss_each = max(6, min(25, max(4, source_size // 2)))
    rss_queries = _rss_queries()
    if len(rss_queries) > 1:
        rss_each = max(5, min(14, rss_each))

    for rss_query in rss_queries:
        _append_google(rss_each, query_text=rss_query)
        _append_bing(rss_each, query_text=rss_query)

    if provider == "hybrid" and len(docs) < source_size:
        gdelt_supplement = max(6, min(20, source_size - len(docs)))
        _append_gdelt(gdelt_supplement, timespan=_history_gdelt_timespan("3months"))

    rss_deduped = _dedupe_news_candidates(cfg, docs, limit=max(8, min(32, source_size)))

    if (
        provider == "rss"
        and cfg.source_type == "news"
        and cfg.domain == "history"
        and _HISTORY_NEWS_RSS_GDELT_SUPPLEMENT
        and len(rss_deduped) < _HISTORY_NEWS_MIN_RSS_CANDIDATES
    ):
        gdelt_supplement = max(
            6,
            min(40, max(_HISTORY_NEWS_GDELT_SUPPLEMENT_SIZE, source_size)),
        )
        _append_gdelt(gdelt_supplement, timespan=_history_gdelt_timespan("12months"))
        return _dedupe_news_candidates(cfg, docs, limit=max(10, min(40, source_size + 8)))

    return rss_deduped


def _save_raw_checkpoint(
    base_dir: Optional[Path],
    cfg: CollectionConfig,
    raw_docs: List[Dict],
    checkpoint_prefix_docs: Optional[List[Dict]] = None,
) -> None:
    if base_dir is None or (not raw_docs):
        return
    raw_dir = base_dir / "raw" / cfg.name
    try:
        merged = list(checkpoint_prefix_docs or []) + list(raw_docs)

        # Keep first-seen order, avoid duplicate writes when URL repeats across attempts.
        deduped: List[Dict] = []
        seen_keys: Set[str] = set()
        for row in merged:
            if not isinstance(row, dict):
                continue
            row_source_type = (row.get("source_type") or cfg.source_type or "").strip().lower()
            url_key = _url_key_for_source_type(row.get("url", ""), row_source_type)
            if url_key:
                key = f"url::{url_key}"
            else:
                key = f"id::{row.get('id') or ''}::{row.get('title') or ''}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(row)

        save_json(raw_dir / "raw_documents.json", deduped)
        save_csv(raw_dir / "raw_documents.csv", deduped)
    except Exception as err:
        logging.warning("%s raw checkpoint save failed error=%s", cfg.name, err)


def collect_data(
    session: requests.Session,
    browser: Optional[Any],
    cfg: CollectionConfig,
    target_max: int,
    news_provider: str,
    max_rounds: int,
    pubmed_date_range: Optional[tuple[str, str]] = None,
    collection_attempt: int = 1,
    base_dir: Optional[Path] = None,
    raw_checkpoint_every: int = _RAW_CHECKPOINT_EVERY,
    checkpoint_prefix_docs: Optional[List[Dict]] = None,
    seed_seen_urls: Optional[Set[str]] = None,
) -> List[Dict]:
    raw_docs: List[Dict] = []
    seen_urls: Set[str] = set()
    seen_news_title_keys: Set[str] = set()
    collection_browser = _browser_for_collection(cfg, browser)
    academic_wikipedia_cooldown_until = 0.0
    academic_wikipedia_429_streak = 0
    for existing_url in (seed_seen_urls or set()):
        key = _url_key_for_source_type(existing_url, cfg.source_type)
        if key:
            seen_urls.add(key)
    if cfg.source_type == "news":
        for row in (checkpoint_prefix_docs or []):
            if not isinstance(row, dict):
                continue
            title_key = normalize_text(clean_text(row.get("title", "")))
            if title_key:
                seen_news_title_keys.add(title_key)
    last_raw_checkpoint_size = 0
    docs_seeded_for_collection = False
    docs_seed_cursor = max(0, (collection_attempt - 1) * 37)
    stalled_rounds = 0

    for round_num in range(1, max_rounds + 1):
        effective_round = max(1, round_num + max(0, collection_attempt - 1))
        keyword_batch = expand_keywords(cfg.keywords, cfg.domain, effective_round, cfg.source_type)
        round_start_count = len(raw_docs)
        no_growth_keywords = 0
        no_growth_keyword_limit = _keyword_no_growth_limit(cfg, len(keyword_batch))
        news_wait_min_ms, news_wait_max_ms = _news_scrape_wait_bounds(cfg)
        logging.info("%s round=%s keywords=%s", cfg.name, effective_round, len(keyword_batch))
        if cfg.source_type == "news":
            if cfg.domain == "history":
                source_size = max(18, min(40, max(20, target_max // 22)))
            else:
                source_size = max(10, min(24, max(12, target_max // 30)))
            logging.info(
                "%s news_scrape_settings source_size=%s wait_ms=%s-%s progress_every=%s",
                cfg.name,
                source_size,
                news_wait_min_ms,
                news_wait_max_ms,
                _NEWS_SCRAPE_PROGRESS_LOG_EVERY,
            )
        elif cfg.source_type == "technical":
            source_size = _technical_source_size(cfg, target_max)
            if cfg.domain == "history":
                logging.info(
                    "%s history_technical_settings source_size=%s primary=%s",
                    cfg.name,
                    source_size,
                    cfg.primary_source,
                )
        elif cfg.source_type == "academic":
            source_size = _academic_source_size(cfg, target_max)
            logging.info(
                "%s academic_source_settings source_size=%s primary=%s",
                cfg.name,
                source_size,
                cfg.primary_source,
            )
        else:
            source_size = max(20, min(100, target_max * 10))

        for keyword_index, keyword in enumerate(keyword_batch):
            if len(raw_docs) >= target_max * 3:
                break

            before_keyword_count = len(raw_docs)
            wikipedia_blocked_for_keyword = False

            new_docs: List[Dict] = []
            try:
                # Advance source pagination per collection attempt to avoid repeatedly fetching page/start=0.
                start_offset = max(0, (collection_attempt - 1) * source_size)
                if cfg.primary_source == "pubmed" and pubmed_date_range:
                    offset_stride = max(1, source_size // 2)
                    offset_window = max(120, source_size * 10)
                    start_offset += (keyword_index * offset_stride) % offset_window
                if cfg.primary_source == "wikipedia" and cfg.source_type == "academic" and cfg.domain == "history":
                    # Rotate search offsets to reduce repeated top-page duplicates across keywords.
                    offset_stride = max(1, source_size // 2)
                    offset_window = max(120, source_size * 12)
                    start_offset += ((keyword_index * offset_stride) % offset_window)
                if cfg.primary_source == "arxiv":
                    new_docs = parse_arxiv(session, cfg, keyword, start=start_offset, size=source_size)
                elif cfg.primary_source == "pubmed":
                    mindate = pubmed_date_range[0] if pubmed_date_range else None
                    maxdate = pubmed_date_range[1] if pubmed_date_range else None
                    for page in range(_PUBMED_PAGES_PER_KEYWORD):
                        page_start = start_offset + (page * source_size)
                        page_docs = parse_pubmed(
                            session,
                            cfg,
                            keyword,
                            start=page_start,
                            size=source_size,
                            mindate=mindate,
                            maxdate=maxdate,
                            datetype="pdat",
                        )
                        if not page_docs:
                            break
                        new_docs.extend(page_docs)
                elif cfg.primary_source == "wikipedia":
                    if cfg.source_type == "technical" and cfg.domain == "technology":
                        if not docs_seeded_for_collection:
                            seed_target = max(30, min(80, max(40, target_max // 20)))
                            seed_docs = collect_technology_documentation_seeds(
                                session=session,
                                browser=collection_browser,
                                cfg=cfg,
                                max_results=seed_target,
                                seed_offset=docs_seed_cursor,
                                exclude_urls=seen_urls,
                            )
                            new_docs.extend(seed_docs)
                            docs_seed_cursor += max(1, seed_target)
                            logging.info("%s docs_seeded=%s", cfg.name, len(seed_docs))
                            docs_seeded_for_collection = True

                        docs_target = max(8, min(20, max(6, source_size // 2)))
                        web_docs = scrape_web_search(collection_browser, cfg, keyword, max_results=docs_target, session=session)
                        new_docs.extend(web_docs)

                        # If search results are sparse/noisy, refill from additional trusted docs seeds.
                        if len(web_docs) < max(4, docs_target // 3):
                            refill_target = max(8, min(24, docs_target + 4))
                            refill_docs = collect_technology_documentation_seeds(
                                session=session,
                                browser=collection_browser,
                                cfg=cfg,
                                max_results=refill_target,
                                seed_offset=docs_seed_cursor + (keyword_index * 17),
                                exclude_urls=seen_urls,
                            )
                            new_docs.extend(refill_docs)
                            docs_seed_cursor += max(1, refill_target)
                            if refill_docs:
                                logging.info("%s docs_refill keyword=%s added=%s", cfg.name, keyword, len(refill_docs))
                    else:
                        if cfg.source_type == "academic" and cfg.domain == "history":
                            now_ts = time.time()
                            if now_ts < academic_wikipedia_cooldown_until:
                                wikipedia_blocked_for_keyword = True
                                remaining = int(max(1, academic_wikipedia_cooldown_until - now_ts))
                                logging.info(
                                    "%s wikipedia cooldown active; skipping wiki fetch for keyword=%s (%ss remaining)",
                                    cfg.name,
                                    keyword,
                                    remaining,
                                )
                            else:
                                try:
                                    new_docs = parse_wikipedia(session, cfg, keyword, offset=start_offset, size=min(50, source_size))
                                    academic_wikipedia_429_streak = 0
                                except Exception as wiki_err:
                                    err_text = str(wiki_err).lower()
                                    if "429" in err_text and "wikipedia" in err_text:
                                        wikipedia_blocked_for_keyword = True
                                        academic_wikipedia_429_streak += 1
                                        if academic_wikipedia_429_streak >= _ACADEMIC_WIKIPEDIA_MAX_429_STREAK:
                                            academic_wikipedia_cooldown_until = time.time() + _ACADEMIC_WIKIPEDIA_COOLDOWN_SECONDS
                                            logging.warning(
                                                "%s hit repeated Wikipedia 429 (%s). Cooling down Wikipedia fetches for %ss",
                                                cfg.name,
                                                academic_wikipedia_429_streak,
                                                _ACADEMIC_WIKIPEDIA_COOLDOWN_SECONDS,
                                            )
                                        else:
                                            logging.warning(
                                                "%s Wikipedia 429 detected for keyword=%s (streak=%s/%s)",
                                                cfg.name,
                                                keyword,
                                                academic_wikipedia_429_streak,
                                                _ACADEMIC_WIKIPEDIA_MAX_429_STREAK,
                                            )
                                    else:
                                        raise
                        else:
                            new_docs = parse_wikipedia(session, cfg, keyword, offset=start_offset, size=min(50, source_size))
                elif cfg.primary_source == "news":
                    new_docs = _collect_news_docs(
                        session=session,
                        cfg=cfg,
                        keyword=keyword,
                        source_size=source_size,
                        news_provider=news_provider,
                    )

                # Supplemental sources improve coverage without changing existing primary-source behavior.
                if cfg.source_type == "academic":
                    crossref_size = max(6, min(20, source_size // 3))
                    crossref_stride = 1
                    if cfg.primary_source == "wikipedia":
                        # For history_academic (Wikipedia primary), call Crossref less frequently to avoid 429 bursts.
                        crossref_stride = max(2, _env_int("RAG_ACADEMIC_CROSSREF_EVERY_N_KEYWORDS", 3))
                        crossref_size = max(6, min(12, crossref_size))
                        if cfg.domain == "history":
                            # History-academic needs broader non-Wikipedia coverage to avoid early plateaus.
                            crossref_stride = max(1, _env_int("RAG_HISTORY_ACADEMIC_CROSSREF_EVERY_N_KEYWORDS", 1))
                            crossref_size = max(crossref_size, max(10, min(18, source_size)))
                        if wikipedia_blocked_for_keyword:
                            # Keep collection moving during Wikipedia cooldown periods.
                            crossref_stride = 1
                            crossref_size = max(crossref_size, 10)
                    if keyword_index % crossref_stride == 0:
                        new_docs.extend(parse_crossref(session, cfg, keyword, size=crossref_size))
            except Exception as err:
                logging.warning("API collect failed for %s keyword=%s error=%s", cfg.name, keyword, err)

            enriched_docs: List[Dict] = []
            docs_for_enrichment: List[Dict] = new_docs

            if cfg.source_type == "news" and new_docs:
                deduped_candidates: List[Dict] = []
                keyword_seen_urls: Set[str] = set()
                keyword_seen_titles: Set[str] = set()

                for candidate in new_docs:
                    candidate_url = (candidate.get("url") or "").strip()
                    if not candidate_url:
                        continue

                    candidate_url_key = _url_key_for_source_type(candidate_url, "news")
                    if not candidate_url_key:
                        continue
                    if candidate_url_key in seen_urls or candidate_url_key in keyword_seen_urls:
                        continue

                    candidate_title = clean_text(candidate.get("title", ""))
                    candidate_text = clean_text(candidate.get("text", ""))
                    candidate_title_key = normalize_text(candidate_title)
                    if candidate_title_key and candidate_title_key in keyword_seen_titles:
                        continue
                    if candidate_title_key and candidate_title_key in seen_news_title_keys:
                        continue

                    # Use fast title/summary checks first to skip obvious off-category links.
                    if candidate_text:
                        if not is_news_doc_domain_relevant(
                            title=candidate_title,
                            text=candidate_text,
                            domain=cfg.domain,
                            keywords=cfg.keywords,
                            min_hits=1,
                        ):
                            continue

                    row = dict(candidate)
                    if candidate_title:
                        row["title"] = candidate_title
                    if candidate_text:
                        row["text"] = candidate_text

                    keyword_seen_urls.add(candidate_url_key)
                    if candidate_title_key:
                        keyword_seen_titles.add(candidate_title_key)
                    deduped_candidates.append(row)

                docs_for_enrichment = deduped_candidates
                logging.info("%s keyword=%s scrape_links=%s", cfg.name, keyword, len(docs_for_enrichment))

                if (
                    cfg.domain == "history"
                    and _HISTORY_NEWS_SPARSE_WEB_SUPPLEMENT
                    and len(docs_for_enrichment) < _HISTORY_NEWS_MIN_RSS_CANDIDATES
                ):
                    supplement_target = max(
                        4,
                        min(18, max(_HISTORY_NEWS_WEB_SUPPLEMENT_RESULTS, _HISTORY_NEWS_MIN_RSS_CANDIDATES)),
                    )
                    web_supplement = scrape_web_search(
                        collection_browser,
                        cfg,
                        keyword,
                        max_results=supplement_target,
                        session=session,
                    )
                    if web_supplement:
                        docs_for_enrichment = _dedupe_news_candidates(
                            cfg,
                            docs_for_enrichment + web_supplement,
                            limit=max(8, len(docs_for_enrichment) + supplement_target),
                        )
                        logging.info(
                            "%s keyword=%s sparse_rss_supplement web_added=%s scrape_links=%s",
                            cfg.name,
                            keyword,
                            len(web_supplement),
                            len(docs_for_enrichment),
                        )

            for doc_idx, doc in enumerate(docs_for_enrichment, start=1):
                if cfg.source_type == "news":
                    if (
                        doc_idx == 1
                        or doc_idx % _NEWS_SCRAPE_PROGRESS_LOG_EVERY == 0
                        or doc_idx == len(docs_for_enrichment)
                    ):
                        logging.info(
                            "%s keyword=%s scrape_progress=%s/%s",
                            cfg.name,
                            keyword,
                            doc_idx,
                            len(docs_for_enrichment),
                        )

                    url = (doc.get("url") or "").strip()
                    if not url:
                        continue

                    title_r = ""
                    text_r = ""
                    date_r = ""
                    final_url = url

                    title_r, text_r, date_r, final_url = fetch_article_text_requests_with_url(
                        session,
                        url,
                        timeout=14,
                        retries=1,
                        interstitial_retry=False,
                    )

                    if (not text_r) and collection_browser is not None:
                        wait_ms = random.randint(news_wait_min_ms, news_wait_max_ms)
                        title_r, text_r, date_r, final_url = fetch_article_text_playwright_with_url(
                            collection_browser,
                            final_url or url,
                            post_load_wait_ms=wait_ms,
                        )

                    if final_url:
                        doc["url"] = final_url

                    final_url_key = _url_key_for_source_type(doc.get("url", ""), "news")
                    if final_url_key and final_url_key in seen_urls:
                        continue

                    candidate_title_key = normalize_text(clean_text(doc.get("title", "")))
                    if candidate_title_key and candidate_title_key in seen_news_title_keys:
                        continue

                    existing_title = clean_text(doc.get("title", ""))
                    existing_text = clean_text(doc.get("text", ""))

                    # If scraping still fails, skip low-quality/redirect-only news rows.
                    if not text_r:
                        allow_history_snippet_fallback = (
                            cfg.domain == "history"
                            and len(existing_text) >= _HISTORY_NEWS_SNIPPET_FALLBACK_MIN_CHARS
                            and is_news_doc_domain_relevant(
                                title=existing_title,
                                text=existing_text,
                                domain=cfg.domain,
                                keywords=cfg.keywords,
                                min_hits=1,
                            )
                            and not looks_like_access_interstitial(
                                title=existing_title,
                                text=existing_text,
                                url=doc.get("url", ""),
                            )
                        )
                        if allow_history_snippet_fallback:
                            if existing_title:
                                doc["title"] = existing_title
                            doc["text"] = existing_text
                        else:
                            continue
                    else:
                        doc["text"] = text_r
                        if title_r:
                            doc["title"] = title_r
                        if date_r and (not doc.get("publication_date")):
                            doc["publication_date"] = date_r

                    final_title_key = normalize_text(clean_text(doc.get("title", "")))
                    enriched_docs.append(doc)
                    if final_title_key:
                        seen_news_title_keys.add(final_title_key)
                    continue

                if _should_enrich_short_doc(cfg, doc, min_len=300):
                    url = doc.get("url", "")
                    title_r, text_r, date_r = fetch_article_text_requests(session, url)
                    if (not text_r) and collection_browser is not None:
                        title_r, text_r, date_r = fetch_article_text_playwright(collection_browser, url)
                    # If scraping returns very short/empty content, skip this item and move on.
                    if not text_r:
                        continue
                    if text_r:
                        doc["text"] = text_r
                    if title_r:
                        doc["title"] = title_r
                    if date_r and (not doc.get("publication_date")):
                        doc["publication_date"] = date_r
                enriched_docs.append(doc)

            if not enriched_docs:
                if cfg.source_type == "news":
                    # Avoid expensive web-search fallback on every keyword; RSS/GDELT already covers news.
                    if not new_docs:
                        fallback_results = 8
                        if cfg.domain == "history":
                            fallback_results = max(12, _HISTORY_NEWS_WEB_SUPPLEMENT_RESULTS)
                        enriched_docs = scrape_web_search(
                            collection_browser,
                            cfg,
                            keyword,
                            max_results=fallback_results,
                            session=session,
                        )
                elif cfg.source_type == "academic":
                    if _ACADEMIC_ALLOW_WEB_SEARCH_FALLBACK:
                        enriched_docs = scrape_web_search(collection_browser, cfg, keyword, max_results=8, session=session)
                else:
                    enriched_docs = scrape_web_search(collection_browser, cfg, keyword, max_results=20, session=session)

            for doc in enriched_docs:
                if looks_like_access_interstitial(
                    title=doc.get("title", ""),
                    text=doc.get("text", ""),
                    url=doc.get("url", ""),
                ):
                    continue

                if cfg.source_type == "news":
                    if not is_news_doc_domain_relevant(
                        title=doc.get("title", ""),
                        text=doc.get("text", ""),
                        domain=cfg.domain,
                        keywords=cfg.keywords,
                        min_hits=2,
                    ):
                        continue
                if len(raw_docs) >= target_max * 3:
                    break
                url_key = _url_key_for_source_type(doc.get("url", ""), cfg.source_type)
                if url_key and url_key in seen_urls:
                    continue
                if url_key:
                    seen_urls.add(url_key)
                if cfg.source_type == "news":
                    title_key = normalize_text(clean_text(doc.get("title", "")))
                    if title_key:
                        seen_news_title_keys.add(title_key)
                raw_docs.append(doc)

            logging.info("%s raw_progress=%s", cfg.name, len(raw_docs))

            should_checkpoint = False
            if len(raw_docs) > 0 and last_raw_checkpoint_size == 0:
                should_checkpoint = True
            elif raw_checkpoint_every > 0 and (len(raw_docs) - last_raw_checkpoint_size) >= raw_checkpoint_every:
                should_checkpoint = True

            if should_checkpoint:
                _save_raw_checkpoint(
                    base_dir,
                    cfg,
                    raw_docs,
                    checkpoint_prefix_docs=checkpoint_prefix_docs,
                )
                last_raw_checkpoint_size = len(raw_docs)

            added_this_keyword = len(raw_docs) - before_keyword_count
            if added_this_keyword <= 0:
                no_growth_keywords += 1
            else:
                no_growth_keywords = 0

            if no_growth_keywords >= no_growth_keyword_limit:
                logging.info(
                    "%s no-growth plateau: %s consecutive keywords with raw_progress unchanged in round=%s; moving to next round",
                    cfg.name,
                    no_growth_keywords,
                    effective_round,
                )
                break

        round_added = len(raw_docs) - round_start_count
        if round_added <= 0:
            stalled_rounds += 1
            logging.info(
                "%s no new raw docs in round=%s (stalled_rounds=%s/%s)",
                cfg.name,
                effective_round,
                stalled_rounds,
                _NO_GROWTH_STALLED_ROUNDS,
            )
        else:
            stalled_rounds = 0

        if stalled_rounds >= _NO_GROWTH_STALLED_ROUNDS:
            logging.info(
                "%s stopping early after %s stalled rounds with no raw growth",
                cfg.name,
                stalled_rounds,
            )
            break

        if len(raw_docs) >= target_max * 2:
            break

    return raw_docs


def save_data(base_dir: Path, cfg: CollectionConfig, raw_docs: List[Dict], cleaned_docs: List[Dict], final_docs: List[Dict]) -> None:
    raw_dir = base_dir / "raw" / cfg.name
    cleaned_dir = base_dir / "cleaned" / cfg.name
    final_dir = base_dir / "final" / cfg.name

    save_json(raw_dir / "raw_documents.json", raw_docs)
    save_csv(raw_dir / "raw_documents.csv", raw_docs)

    save_json(cleaned_dir / "cleaned_documents.json", cleaned_docs)
    save_csv(cleaned_dir / "cleaned_documents.csv", cleaned_docs)

    save_json(final_dir / "final_documents.json", final_docs)
    save_csv(final_dir / "final_documents.csv", final_docs)


def _age_days(publication_date: str) -> Optional[float]:
    dt = parse_date(publication_date)
    if not dt:
        return None
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)


def _count_docs_in_age_window(
    docs: List[Dict],
    min_age_days: int,
    max_age_days: Optional[int],
) -> int:
    count = 0
    for doc in docs:
        age = _age_days(doc.get("publication_date", ""))
        if age is None:
            continue
        if age < min_age_days:
            continue
        if max_age_days is not None and age > max_age_days:
            continue
        count += 1
    return count


def _prioritize_docs_by_age_window(
    docs: List[Dict],
    min_age_days: int,
    max_age_days: Optional[int],
    strict: bool = True,
) -> List[Dict]:
    in_window: List[Dict] = []
    out_window: List[Dict] = []
    for doc in docs:
        age = _age_days(doc.get("publication_date", ""))
        if age is None:
            out_window.append(doc)
            continue
        if age < min_age_days:
            out_window.append(doc)
            continue
        if max_age_days is not None and age > max_age_days:
            out_window.append(doc)
            continue
        in_window.append(doc)
    if strict:
        return in_window
    return in_window + out_window


def _pubmed_date_range(min_age_days: int, max_age_days: int) -> Optional[tuple[str, str]]:
    if max_age_days <= 0:
        return None
    if max_age_days < min_age_days:
        min_age_days, max_age_days = max_age_days, min_age_days
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=max_age_days)
    end = now - timedelta(days=min_age_days)
    return start.strftime("%Y/%m/%d"), end.strftime("%Y/%m/%d")


_ADJUSTED_WINDOW_BASELINE_DOMAIN = "technology"


def _is_adjusted_condition(condition: ConditionConfig) -> bool:
    return "adj" in (condition.freshness_label or "").lower()


def _effective_age_days(age_days: float, condition: ConditionConfig) -> float:
    if not _is_adjusted_condition(condition):
        return age_days
    domain = (condition.domain or "").strip().lower()
    baseline = _ADJUSTED_WINDOW_BASELINE_DOMAIN
    domain_volatility = DOMAIN_VOLATILITY_DAYS.get(domain)
    baseline_volatility = DOMAIN_VOLATILITY_DAYS.get(baseline)
    if not domain_volatility or not baseline_volatility:
        return age_days
    factor = domain_volatility / float(baseline_volatility)
    return age_days * factor


def _matches_condition_window(age_days: float, condition: ConditionConfig) -> bool:
    effective_age = _effective_age_days(age_days, condition)
    if effective_age < condition.min_age_days:
        return False
    if condition.max_age_days is not None and effective_age > condition.max_age_days:
        return False
    return True


def build_condition_datasets(
    output_dir: Path,
    cleaned_by_collection: Dict[str, List[Dict]],
    target_max: int,
) -> Dict[str, List[Dict]]:
    by_domain_source: Dict[tuple[str, str], List[Dict]] = {}
    for cfg in COLLECTIONS:
        rows = cleaned_by_collection.get(cfg.name, [])
        by_domain_source[(cfg.domain, cfg.source_type)] = rows

    condition_docs: Dict[str, List[Dict]] = {}
    combined: List[Dict] = []

    for condition in RQ1_CONDITIONS:
        candidates: List[Dict] = []
        for source_type in condition.source_mix:
            candidates.extend(by_domain_source.get((condition.domain, source_type), []))

        selected: List[Dict] = []
        seen_keys: Set[str] = set()

        for doc in candidates:
            age = _age_days(doc.get("publication_date", ""))
            if age is None or not _matches_condition_window(age, condition):
                continue

            url_key = (doc.get("url") or "").strip().lower()
            key = url_key or f"{doc.get('title','').strip().lower()}::{doc.get('publication_date','')}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            row = dict(doc)
            score, label = compute_freshness(row.get("publication_date", ""), condition.domain)
            row["freshness_score"] = score
            row["freshness_label"] = label
            row["condition_id"] = condition.condition_id
            row["condition_domain"] = condition.domain
            row["condition_freshness_window"] = condition.freshness_label
            row["condition_source_configuration"] = condition.source_configuration
            selected.append(row)

            if len(selected) >= target_max:
                break

        selected, selected_quality_stats = _quality_filter_docs(selected, cfg=None)
        selected_quality_removed = _quality_drop_count(selected_quality_stats)
        if selected_quality_removed:
            logging.info(
                "Condition %s quality filter removed=%s details=%s",
                condition.condition_id,
                selected_quality_removed,
                selected_quality_stats,
            )

        condition_docs[condition.condition_id] = selected
        combined.extend(selected)

        cond_dir = output_dir / "final" / "conditions" / condition.condition_id
        save_json(cond_dir / "condition_documents.json", selected)
        save_csv(cond_dir / "condition_documents.csv", selected)

    combined, combined_quality_stats = _quality_filter_docs(combined, cfg=None)
    combined_quality_removed = _quality_drop_count(combined_quality_stats)
    if combined_quality_removed:
        logging.info(
            "Conditions combined quality filter removed=%s details=%s",
            combined_quality_removed,
            combined_quality_stats,
        )

    save_json(output_dir / "final" / "conditions" / "combined_conditions_dataset.json", combined)
    save_csv(output_dir / "final" / "conditions" / "combined_conditions_dataset.csv", combined)
    return condition_docs


def _copy_technology_academic_pdfs(base_dir: Path, condition_docs: Dict[str, List[Dict]]) -> None:
    source_dir = base_dir / "raw" / "pdfs" / "technology_academic"
    if not source_dir.exists():
        return

    final_dir = base_dir / "final" / "pdfs" / "technology_academic"
    final_dir.mkdir(parents=True, exist_ok=True)

    copied_total = 0
    for pdf_path in source_dir.glob("*.pdf"):
        target = final_dir / pdf_path.name
        if target.exists():
            continue
        shutil.copy2(pdf_path, target)
        copied_total += 1

    copied_conditions = 0
    for condition_id, docs in condition_docs.items():
        cond_dir = base_dir / "final" / "conditions" / condition_id / "pdfs"
        for doc in docs:
            pdf_name = (doc.get("pdf_file_name") or "").strip()
            if not pdf_name:
                continue
            source_path = source_dir / pdf_name
            if not source_path.exists():
                continue
            cond_dir.mkdir(parents=True, exist_ok=True)
            target_path = cond_dir / pdf_name
            if target_path.exists():
                continue
            shutil.copy2(source_path, target_path)
            copied_conditions += 1

    if copied_total or copied_conditions:
        logging.info(
            "Copied technology_academic PDFs: final=%s condition_copies=%s",
            copied_total,
            copied_conditions,
        )


def process_collection(
    session: requests.Session,
    browser: Optional[Any],
    base_dir: Path,
    cfg: CollectionConfig,
    target_min: int,
    target_max: int,
    max_rounds: int,
    news_provider: str,
    strict: bool,
    initial_raw: Optional[List[Dict]] = None,
    pubmed_date_range: Optional[tuple[str, str]] = None,
    force_collect: bool = False,
) -> List[Dict]:
    all_raw: List[Dict] = list(initial_raw or [])
    final_docs: List[Dict] = []
    strict_min_required = _strict_min_required(cfg, target_min)
    effective_rounds = max_rounds
    attempt = 1

    def _recompute_and_save(all_raw_docs: List[Dict]) -> tuple[List[Dict], int, int, Dict[str, int]]:
        cleaned_docs = clean_data(all_raw_docs, cfg)
        if force_collect and pubmed_date_range and cfg.name == "healthcare_academic":
            cleaned_docs = _prioritize_docs_by_age_window(
                cleaned_docs,
                _HEALTHCARE_ACADEMIC_PUBMED_MIN_AGE_DAYS,
                _HEALTHCARE_ACADEMIC_PUBMED_MAX_AGE_DAYS,
                strict=True,
            )
        deduped_docs, removed_count = deduplicate_data(cleaned_docs, threshold=0.95)
        computed_final = add_final_fields(deduped_docs, cfg)
        computed_final, quality_stats = _quality_filter_docs(computed_final, cfg=cfg)
        quality_removed = _quality_drop_count(quality_stats)
        if quality_removed:
            logging.info("%s final quality filter removed=%s details=%s", cfg.name, quality_removed, quality_stats)

        if len(computed_final) > target_max:
            computed_final = computed_final[:target_max]

        save_data(base_dir, cfg, all_raw_docs, cleaned_docs, computed_final)
        return computed_final, len(cleaned_docs), removed_count, quality_stats

    # If previous RAW docs exist, process them first before fetching more.
    if all_raw:
        final_docs, cleaned_count, removed_count, _quality_stats = _recompute_and_save(all_raw)

        raw_pct = min(100.0, (len(all_raw) / max(1, target_max * 3)) * 100.0)
        cleaned_pct = min(100.0, (cleaned_count / max(1, target_max * 2)) * 100.0)
        final_pct = min(100.0, (len(final_docs) / max(1, target_min)) * 100.0)
        logging.info(
            "%s initial pass raw=%s (%.1f%%) cleaned=%s (%.1f%%) final=%s (%.1f%%) duplicates_removed=%s",
            cfg.name,
            len(all_raw),
            raw_pct,
            cleaned_count,
            cleaned_pct,
            len(final_docs),
            final_pct,
            removed_count,
        )

        if len(final_docs) >= target_min and not force_collect:
            return final_docs
        if force_collect:
            logging.info("%s force_collect enabled; continuing to backfill", cfg.name)

    while attempt <= effective_rounds:
        seed_seen_urls = {
            (d.get("url") or "").strip().lower()
            for d in all_raw
            if isinstance(d, dict) and (d.get("url") or "").strip()
        }

        collected = collect_data(
            session=session,
            browser=browser,
            cfg=cfg,
            target_max=target_max,
            news_provider=news_provider,
            max_rounds=1,
            pubmed_date_range=pubmed_date_range,
            collection_attempt=attempt,
            base_dir=base_dir,
            checkpoint_prefix_docs=all_raw,
            seed_seen_urls=seed_seen_urls,
        )
        download_arxiv_pdfs(
            session=session,
            base_dir=base_dir,
            cfg=cfg,
            docs=collected,
            max_downloads=None,
        )
        all_raw.extend(collected)

        final_docs, cleaned_count, removed_count, _quality_stats = _recompute_and_save(all_raw)

        raw_pct = min(100.0, (len(all_raw) / max(1, target_max * 3)) * 100.0)
        cleaned_pct = min(100.0, (cleaned_count / max(1, target_max * 2)) * 100.0)
        final_pct = min(100.0, (len(final_docs) / max(1, target_min)) * 100.0)

        logging.info(
            "%s attempt=%s raw=%s (%.1f%%) cleaned=%s (%.1f%%) final=%s (%.1f%%) duplicates_removed=%s",
            cfg.name,
            attempt,
            len(all_raw),
            raw_pct,
            cleaned_count,
            cleaned_pct,
            len(final_docs),
            final_pct,
            removed_count,
        )

        if len(final_docs) >= target_min and not force_collect:
            break

        # arXiv can be heavily rate-limited; allow a few extra rounds before strict failure.
        if strict and cfg.primary_source == "arxiv" and attempt == effective_rounds and len(final_docs) < target_min:
            extra = min(4, max(0, target_min // 200))
            if extra > 0:
                effective_rounds += extra
                logging.info(
                    "%s below target after %s rounds (final=%s, target_min=%s). Extending rounds to %s.",
                    cfg.name,
                    attempt,
                    len(final_docs),
                    target_min,
                    effective_rounds,
                )

        attempt += 1

    if strict and len(final_docs) < strict_min_required:
        raise RuntimeError(
            f"Collection {cfg.name} did not reach strict minimum={strict_min_required} "
            f"(target_min={target_min}). final={len(final_docs)}. "
            "Try increasing rounds, improving connectivity/proxies, or broadening keywords."
        )

    if strict and len(final_docs) < target_min:
        logging.warning(
            "%s completed below target_min=%s with final=%s; strict floor=%s accepted.",
            cfg.name,
            target_min,
            len(final_docs),
            strict_min_required,
        )

    return final_docs


def build_dataset(
    output_dir: Path,
    target_min: int,
    target_max: int,
    news_provider: str,
    strict: bool,
    max_rounds: int,
    fresh: bool,
    collection: str,
    on_existing: Literal["auto", "ask", "skip", "recreate", "fill"],
    rq1_query_alignment: bool,
) -> None:
    load_dotenv()

    if fresh and output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)

    ensure_stage_structure(output_dir)
    setup_logging(output_dir)

    session = requests.Session()
    session.headers.update({"User-Agent": "RAG-RQ1-Builder/2.1"})

    playwright_ctx = None
    browser: Optional[Any] = None
    if sync_playwright is not None:
        try:
            playwright_ctx = sync_playwright().start()
            browser = playwright_ctx.chromium.launch(headless=True)
        except Exception as err:
            logging.warning("Playwright unavailable, scraping fallback disabled: %s", err)
            browser = None

    all_final: Dict[str, List[Dict]] = {}
    all_cleaned: Dict[str, List[Dict]] = {}
    combined: List[Dict] = []

    selected_collections = COLLECTIONS if collection == "all" else [cfg for cfg in COLLECTIONS if cfg.name == collection]

    if rq1_query_alignment:
        aligned: List[CollectionConfig] = []
        for cfg in selected_collections:
            domain_queries = RQ1_DOMAIN_QUERIES.get(cfg.domain)
            if domain_queries:
                merged_keywords = list(dict.fromkeys(list(cfg.keywords) + list(domain_queries)))
                aligned.append(
                    CollectionConfig(
                        name=cfg.name,
                        domain=cfg.domain,
                        source_type=cfg.source_type,
                        primary_source=cfg.primary_source,
                        keywords=merged_keywords,
                    )
                )
            else:
                aligned.append(cfg)
        selected_collections = aligned
        logging.info(
            "RQ1 query alignment enabled (expected RQ1 experiments=%s, mode=merged-domain-and-collection-keywords)",
            sum(len(queries) for queries in RQ1_DOMAIN_QUERIES.values()),
        )

    if not selected_collections:
        raise ValueError(f"Unknown collection: {collection}")

    def resolve_existing_action(cfg_name: str, existing_count: int, minimum: int) -> Literal["skip", "recreate", "fill"]:
        if on_existing == "auto":
            if existing_count < minimum:
                logging.info(
                    "%s existing=%s below target_min=%s and --on-existing=auto; defaulting to fill",
                    cfg_name,
                    existing_count,
                    minimum,
                )
                return "fill"
            logging.info(
                "%s existing=%s meets target_min=%s and --on-existing=auto; defaulting to skip",
                cfg_name,
                existing_count,
                minimum,
            )
            return "skip"

        if on_existing == "skip":
            return "skip"
        if on_existing == "recreate":
            return "recreate"
        if on_existing == "fill":
            return "fill"

        if not os.isatty(0):
            logging.info("%s existing=%s and --on-existing=ask in non-interactive mode; defaulting to fill", cfg_name, existing_count)
            return "fill"

        print(
            f"Collection '{cfg_name}' already has {existing_count} final docs (target_min={minimum}).\n"
            "Choose action: [f]ill missing, [s]kip, [r]ecreate (default: f): ",
            end="",
            flush=True,
        )
        choice = (input().strip().lower() or "f")[:1]
        if choice == "s":
            return "skip"
        if choice == "r":
            return "recreate"
        return "fill"

    try:
        for cfg in selected_collections:
            raw_path = output_dir / "raw" / cfg.name / "raw_documents.json"
            final_path = output_dir / "final" / cfg.name / "final_documents.json"
            existing_raw = load_json_list(raw_path)
            existing_final = load_json_list(final_path)

            pubmed_date_range: Optional[tuple[str, str]] = None
            force_collect = False
            backfill_needed = False
            if (
                cfg.name == "healthcare_academic"
                and cfg.primary_source == "pubmed"
                and _HEALTHCARE_ACADEMIC_C7_BACKFILL
            ):
                date_range = _pubmed_date_range(
                    _HEALTHCARE_ACADEMIC_PUBMED_MIN_AGE_DAYS,
                    _HEALTHCARE_ACADEMIC_PUBMED_MAX_AGE_DAYS,
                )
                if date_range:
                    window_target_min = max(_HEALTHCARE_ACADEMIC_C7_MIN_DOCS, target_min)
                    existing_window_count = _count_docs_in_age_window(
                        existing_final,
                        _HEALTHCARE_ACADEMIC_PUBMED_MIN_AGE_DAYS,
                        _HEALTHCARE_ACADEMIC_PUBMED_MAX_AGE_DAYS,
                    )
                    # Always set date range if C7 backfill is requested so new queries are scoped
                    pubmed_date_range = date_range
                    force_collect = True
                    if existing_window_count < window_target_min or on_existing == "recreate":
                        backfill_needed = True
                        logging.info(
                            "%s C7 backfill enabled window=%s..%s existing_window=%s target_window_min=%s",
                            cfg.name,
                            date_range[0],
                            date_range[1],
                            existing_window_count,
                            window_target_min,
                        )
                    else:
                        logging.info(
                            "%s C7 backfill met (existing_window=%s >= target_window_min=%s)",
                            cfg.name,
                            existing_window_count,
                            window_target_min,
                        )

            docs: List[Dict]
            if existing_final:
                action = resolve_existing_action(cfg.name, len(existing_final), target_min)
                if backfill_needed and action == "skip" and on_existing in {"auto", "ask"}:
                    logging.info("%s C7 backfill requires fill; overriding skip to fill", cfg.name)
                    action = "fill"
                elif backfill_needed and action == "skip":
                    logging.warning(
                        "%s C7 backfill needed but --on-existing=%s; skipping per request",
                        cfg.name,
                        on_existing,
                    )
                if action == "skip":
                    logging.info("%s skipped because data already exists (%s docs)", cfg.name, len(existing_final))
                    docs = existing_final
                elif action == "recreate":
                    logging.info("%s recreating dataset from scratch", cfg.name)
                    docs = process_collection(
                        session=session,
                        browser=browser,
                        base_dir=output_dir,
                        cfg=cfg,
                        target_min=target_min,
                        target_max=target_max,
                        max_rounds=max_rounds,
                        news_provider=news_provider,
                        strict=strict,
                        pubmed_date_range=pubmed_date_range,
                        force_collect=force_collect,
                        initial_raw=[],
                    )
                else:
                    logging.info("%s filling missing docs: existing=%s target_min=%s", cfg.name, len(existing_final), target_min)
                    docs = process_collection(
                        session=session,
                        browser=browser,
                        base_dir=output_dir,
                        cfg=cfg,
                        target_min=target_min,
                        target_max=target_max,
                        max_rounds=max_rounds,
                        news_provider=news_provider,
                        strict=strict,
                        pubmed_date_range=pubmed_date_range,
                        force_collect=force_collect,
                        initial_raw=existing_raw or existing_final,
                    )
            else:
                docs = process_collection(
                    session=session,
                    browser=browser,
                    base_dir=output_dir,
                    cfg=cfg,
                    target_min=target_min,
                    target_max=target_max,
                    max_rounds=max_rounds,
                    news_provider=news_provider,
                    strict=strict,
                    pubmed_date_range=pubmed_date_range,
                    force_collect=force_collect,
                    initial_raw=[],
                )

            docs, docs_quality_stats = _quality_filter_docs(docs, cfg=cfg)
            docs_quality_removed = _quality_drop_count(docs_quality_stats)
            if docs_quality_removed:
                logging.info("%s output quality filter removed=%s details=%s", cfg.name, docs_quality_removed, docs_quality_stats)
                save_json(final_path, docs)
                save_csv(final_path.with_suffix(".csv"), docs)

            all_final[cfg.name] = docs
            cleaned_path = output_dir / "cleaned" / cfg.name / "cleaned_documents.json"
            all_cleaned[cfg.name] = load_json_list(cleaned_path)
            combined.extend(docs)

        condition_docs = build_condition_datasets(
            output_dir=output_dir,
            cleaned_by_collection=all_cleaned,
            target_max=target_max,
        )
        _copy_technology_academic_pdfs(base_dir=output_dir, condition_docs=condition_docs)
        condition_min_docs = _CONDITION_MIN_DOCS
        condition_shortfalls = {
            condition_id: max(0, condition_min_docs - len(rows))
            for condition_id, rows in condition_docs.items()
        }
        conditions_below_min = {
            condition_id: shortfall
            for condition_id, shortfall in condition_shortfalls.items()
            if shortfall > 0
        }
        if conditions_below_min:
            logging.warning(
                "Conditions below minimum=%s: %s",
                condition_min_docs,
                conditions_below_min,
            )

        combined, combined_quality_stats = _quality_filter_docs(combined, cfg=None)
        combined_quality_removed = _quality_drop_count(combined_quality_stats)
        if combined_quality_removed:
            logging.info("Combined final quality filter removed=%s details=%s", combined_quality_removed, combined_quality_stats)

        save_json(output_dir / "final" / "combined_dataset.json", combined)
        save_csv(output_dir / "final" / "combined_dataset.csv", combined)

        collection_progress: Dict[str, Dict[str, float | int]] = {}
        for name, rows in all_final.items():
            count = len(rows)
            collection_progress[name] = {
                "final_count": count,
                "target_min": target_min,
                "target_max": target_max,
                "progress_to_target_min_pct": round(min(100.0, (count / max(1, target_min)) * 100.0), 2),
                "fill_to_target_max_pct": round(min(100.0, (count / max(1, target_max)) * 100.0), 2),
            }

        summary = {
            "generated_at": now_iso(),
            "target_min": target_min,
            "target_max": target_max,
            "strict": strict,
            "news_provider": news_provider,
            "rq1_query_alignment": rq1_query_alignment,
            "collections": {name: len(rows) for name, rows in all_final.items()},
            "collection_progress": collection_progress,
            "conditions": {name: len(rows) for name, rows in condition_docs.items()},
            "condition_min_docs": condition_min_docs,
            "condition_shortfalls": condition_shortfalls,
            "conditions_meet_minimum": len(conditions_below_min) == 0,
            "expected_conditions": len(RQ1_CONDITIONS),
            "total_documents": len(combined),
        }
        save_json(output_dir / "summary.json", [summary])

        logging.info("Build completed. Summary: %s", json.dumps(summary, indent=2))
    except KeyboardInterrupt:
        logging.warning("Build interrupted by user; partial outputs preserved.")
        return
    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception as err:
                logging.warning("Playwright browser close ignored: %s", err)
        if playwright_ctx is not None:
            try:
                playwright_ctx.stop()
            except Exception as err:
                logging.warning("Playwright context stop ignored: %s", err)
