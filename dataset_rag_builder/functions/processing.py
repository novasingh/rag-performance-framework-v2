from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import DOMAIN_HINTS, DOMAIN_VOLATILITY_DAYS, CollectionConfig
from ..helpers.text_helpers import clean_text, is_english_text, normalize_text, parse_date
from ..utils.io_utils import now_iso
from ..utils.model_utils import get_embedding_model

_EMBED_MODEL: SentenceTransformer | None = None

_OFFTOPIC_NEWS_PATTERNS = [
    r"\bipl\b",
    r"\bcricket\b",
    r"\bfootball\b",
    r"\bsoccer\b",
    r"\bnba\b",
    r"\bnfl\b",
    r"\btennis\b",
    r"\bfifa\b",
    r"\buefa\b",
    r"\bpremier league\b",
    r"\bla liga\b",
    r"\bserie a\b",
    r"\bsuper bowl\b",
    r"\bworld cup\b",
    r"\bmatch highlights\b",
    r"\blive score\b",
    r"\binnings\b",
    r"\bbowler\b",
    r"\bbatsman\b",
    r"\bhat-?trick\b",
]
_OFFTOPIC_NEWS_REGEX = re.compile("|".join(f"(?:{p})" for p in _OFFTOPIC_NEWS_PATTERNS))
_DEFAULT_PUBLISHED_DATE = "2024-03-15T00:00:00+00:00"
_DEFAULT_WORD_COUNT = 312
_FRESHNESS_WINDOWS_BY_DOMAIN = {
    "technology": {
        "w1_under_1week",
        "w2_1week_1month",
        "w3_1to6months",
        "w4_over_6months",
    },
    "healthcare": {
        "w1_1to6months",
        "w2_6to12months",
        "w3_1to3years",
        "w4_over_3years",
    },
    "history": {
        "w1_1to5years",
        "w2_5to10years",
        "w3_over10years",
        "w4_archival",
    },
}


_HISTORY_DOC_URL_HINTS = (
    "/archive",
    "/archives",
    "/collection",
    "/collections",
    "/catalog",
    "/catalogs",
    "/records",
    "/research",
    "/manuscript",
    "/manuscripts",
    "/heritage",
    "/museum",
    "/chronicle",
    "/timeline",
    "/finding-aid",
    "/finding_aid",
    "/preservation",
    "digitalcollections",
    "digital-collections",
    "chroniclingamerica",
    "iiif",
    "openarchives",
    "dublincore",
    "tei-c",
    "repository",
)

_HISTORY_DOC_STRUCTURE_TERMS = {
    "archive",
    "archives",
    "archival",
    "collection",
    "collections",
    "catalog",
    "catalogs",
    "record",
    "records",
    "research",
    "manuscript",
    "manuscripts",
    "finding aid",
    "finding-aid",
    "preservation",
    "digital collection",
    "digital collections",
    "museum",
    "heritage",
    "history",
    "historical",
    "chronicle",
    "timeline",
}
_HEALTHCARE_CORE_TERMS = {
    "healthcare",
    "health",
    "medical",
    "clinical",
    "hospital",
    "patient",
    "disease",
    "treatment",
    "therapy",
    "diagnosis",
    "medicine",
    "public health",
    "care",
}

_HISTORY_CORE_TERMS = {
    "history",
    "historical",
    "archaeology",
    "archaeological",
    "heritage",
    "archive",
    "archival",
    "museum",
    "artifact",
    "artifacts",
    "manuscript",
    "manuscripts",
    "repatriation",
    "antiquities",
    "monument",
    "unesco",
    "excavation",
    "commemoration",
}

_HISTORY_NOISE_TERMS = {
    "live score",
    "match",
    "fixture",
    "premier league",
    "football",
    "soccer",
    "nba",
    "nfl",
    "celebrity",
    "box office",
    "trailer",
    "streaming",
    "episode",
}

_TECH_FINANCE_NOISE_TERMS = {
    "nasdaq",
    "stock",
    "shares",
    "earnings",
    "semiconductor",
    "chip",
    "cloud",
    "startup",
    "venture capital",
    "crypto",
    "bitcoin",
    "blockchain",
}

_TECH_DOC_URL_HINTS = (
    "docs.",
    "/docs/",
    "developer.",
    "/developer/",
    "/api/",
    "readthedocs",
    "kubernetes.io/docs",
    "learn.microsoft.com",
    "developer.mozilla.org",
)

_TECH_DOC_TEXT_HINTS = {
    "documentation",
    "developer",
    "api",
    "reference",
    "tutorial",
    "guide",
    "quickstart",
    "installation",
    "configuration",
    "endpoint",
    "sdk",
    "example",
}

_TECH_DOC_TITLE_HINTS = {
    "documentation",
    "docs",
    "api",
    "reference",
    "developer",
    "guide",
    "tutorial",
    "quickstart",
    "sdk",
    "cli",
}

_TECH_DOC_NOISE_TERMS = {
    "stock",
    "investor",
    "market",
    "newsletter",
    "breaking news",
    "subscribe",
    "earnings",
    "captcha",
    "cookie consent",
    "privacy preference",
}

_INTERSTITIAL_TITLE_HINTS = {
    "just a moment",
    "please wait",
    "attention required",
    "checking your browser",
    "verify you are human",
    "access denied",
    "one more step",
}

_INTERSTITIAL_TEXT_HINTS = {
    "checking if the site connection is secure",
    "enable javascript and cookies to continue",
    "malicious bots",
    "verify you are human",
    "ddos protection by",
    "cloudflare ray id",
    "please stand by while we are checking your browser",
    "captcha",
    "security check",
    "browser integrity check",
}

_INTERSTITIAL_URL_HINTS = (
    "cdn-cgi/challenge",
    "challenge-platform",
    "captcha",
    "turnstile",
)


def compute_freshness(publication_date: str, domain: str) -> Tuple[float, str]:
    dt = parse_date(publication_date)
    if not dt:
        return 0.0, "Low"

    age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
    decay = DOMAIN_VOLATILITY_DAYS[domain]
    score = max(0.0, min(1.0, 1.0 - (age_days / float(decay))))

    if score >= 0.66:
        label = "High"
    elif score >= 0.33:
        label = "Medium"
    else:
        label = "Low"
    return round(score, 4), label


def _canonical_domain(domain: str) -> str:
    low = (domain or "").strip().lower()
    if low == "historical":
        return "history"
    return low


def compute_freshness_window(publication_date: str, domain: str) -> str:
    domain_key = _canonical_domain(domain)
    dt = parse_date(publication_date)
    if not dt:
        if domain_key == "healthcare":
            return "w4_over_3years"
        if domain_key == "history":
            return "w4_archival"
        return "w4_over_6months"

    age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)

    if domain_key == "healthcare":
        if age_days <= 180:
            return "w1_1to6months"
        if age_days <= 365:
            return "w2_6to12months"
        if age_days <= 1095:
            return "w3_1to3years"
        return "w4_over_3years"

    if domain_key == "history":
        if age_days <= 1825:
            return "w1_1to5years"
        if age_days <= 3650:
            return "w2_5to10years"
        if age_days <= 5840:
            return "w3_over10years"
        return "w4_archival"

    # Technology / default high-volatility mapping.
    if age_days <= 7:
        return "w1_under_1week"
    if age_days <= 30:
        return "w2_1week_1month"
    if age_days <= 180:
        return "w3_1to6months"
    return "w4_over_6months"


def _is_valid_freshness_window(window: str, domain: str) -> bool:
    domain_key = _canonical_domain(domain)
    allowed = _FRESHNESS_WINDOWS_BY_DOMAIN.get(domain_key)
    if not allowed:
        return False
    return (window or "").strip() in allowed


def _normalized_published_date(publication_date: str) -> str:
    dt = parse_date(publication_date)
    if dt is not None:
        return dt.isoformat()
    if publication_date:
        return publication_date
    return _DEFAULT_PUBLISHED_DATE


def _infer_source_api(source_name: str, url: str) -> str:
    low_source = (source_name or "").strip().lower()
    low_url = (url or "").strip().lower()

    if "arxiv" in low_source or "arxiv.org" in low_url:
        return "arxiv"
    if "pubmed" in low_source or "pubmed.ncbi.nlm.nih.gov" in low_url:
        return "pubmed"
    if "gdelt" in low_source or "gdeltproject.org" in low_url:
        return "gdelt"
    if "google" in low_source and "news" in low_source:
        return "google_news_rss"
    if "news.google.com" in low_url:
        return "google_news_rss"
    if "bing" in low_source and "news" in low_source:
        return "bing_news_rss"
    if "bing.com/news" in low_url:
        return "bing_news_rss"
    if "wikipedia" in low_source or "wikipedia.org" in low_url:
        return "wikipedia"
    if "crossref" in low_source:
        return "crossref"
    if "googlesearch" in low_source:
        return "google_search"
    if "bingsearch" in low_source:
        return "bing_search"
    if "doi.org" in low_url:
        return "doi"
    return "unknown"


def _word_count(text: str) -> int:
    count = len(re.findall(r"\b\w+\b", text or ""))
    if count > 0:
        return count
    return _DEFAULT_WORD_COUNT


def _term_present(normalized_blob: str, term: str) -> bool:
    pieces = [p for p in (term or "").strip().lower().split() if p]
    if not pieces:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(p) for p in pieces) + r"\b"
    return bool(re.search(pattern, normalized_blob))


def ensure_generated_metadata(
    record: Dict,
    source_name: Optional[str] = None,
    default_domain: Optional[str] = None,
    default_source_type: Optional[str] = None,
    source_api: Optional[str] = None,
) -> Dict:
    row = dict(record)

    if default_domain and (not row.get("domain")):
        row["domain"] = default_domain
    if default_source_type and (not row.get("source_type")):
        row["source_type"] = default_source_type

    publication_date = str(row.get("publication_date") or "").strip()

    if not row.get("published_date"):
        row["published_date"] = _normalized_published_date(publication_date)

    domain_key = str(row.get("domain") or default_domain or "technology")
    existing_window = str(row.get("freshness_window") or "")
    if (not existing_window) or (not _is_valid_freshness_window(existing_window, domain_key)):
        row["freshness_window"] = compute_freshness_window(publication_date, domain_key)

    if not row.get("source_api"):
        inferred = source_api or _infer_source_api(source_name or str(row.get("source_name") or ""), str(row.get("url") or ""))
        row["source_api"] = inferred

    # Keep word_count synchronized with the current text payload.
    row["word_count"] = _word_count(str(row.get("text") or ""))

    if not row.get("collected_at"):
        row["collected_at"] = now_iso()

    if not row.get("language"):
        row["language"] = "en"

    return row


def build_doc(
    domain: str,
    source_type: str,
    source_name: str,
    title: str,
    text: str,
    publication_date: str,
    url: str,
    author: str | None = None,
    source_api: str | None = None,
) -> Dict:
    score, label = compute_freshness(publication_date, domain)
    row = {
        "id": f"{domain}_{source_type}_{uuid.uuid4().hex[:12]}",
        "title": (title or "").strip(),
        "text": (text or "").strip(),
        "source_name": source_name,
        "source_type": source_type,
        "domain": domain,
        "publication_date": publication_date,
        "freshness_score": score,
        "freshness_label": label,
        "url": url,
        "author": author,
    }
    return ensure_generated_metadata(
        row,
        source_name=source_name,
        default_domain=domain,
        default_source_type=source_type,
        source_api=source_api,
    )


def extract_domain_terms(domain: str, keywords: List[str]) -> Set[str]:
    terms = set(DOMAIN_HINTS[domain])
    for phrase in keywords:
        for token in re.split(r"[^a-zA-Z0-9]+", phrase.lower()):
            if len(token) >= 4:
                terms.add(token)
    return terms


def count_domain_term_hits(text: str, domain: str, keywords: List[str]) -> int:
    blob = normalize_text(text)
    terms = extract_domain_terms(domain, keywords)
    return sum(1 for t in terms if t in blob)


def looks_domain_relevant(text: str, domain: str, keywords: List[str], min_hits: int = 2) -> bool:
    hits = count_domain_term_hits(text, domain, keywords)
    return hits >= min_hits


def looks_like_offtopic_news(text: str) -> bool:
    blob = normalize_text(text)
    return bool(_OFFTOPIC_NEWS_REGEX.search(blob))


def is_news_doc_domain_relevant(
    title: str,
    text: str,
    domain: str,
    keywords: List[str],
    min_hits: int = 2,
) -> bool:
    blob = f"{title} {text}"
    normalized_blob = normalize_text(blob)
    if not looks_domain_relevant(blob, domain, keywords, min_hits=min_hits):
        return False

    # Sports pages often contain a token like "health" in site navigation; require extra evidence.
    if looks_like_offtopic_news(blob) and not looks_domain_relevant(blob, domain, keywords, min_hits=min_hits + 1):
        return False

    # Extra guardrail for healthcare news: reject technology/finance-heavy stories with weak clinical evidence.
    if (domain or "").strip().lower() == "healthcare":
        healthcare_hits = sum(1 for term in _HEALTHCARE_CORE_TERMS if _term_present(normalized_blob, term))
        noise_hits = sum(1 for term in _TECH_FINANCE_NOISE_TERMS if _term_present(normalized_blob, term))
        if healthcare_hits < 2:
            return False
        if noise_hits >= 4 and healthcare_hits <= 3:
            return False

    # History news should emphasize archival/heritage signal, not entertainment or sports noise.
    if (domain or "").strip().lower() == "history":
        history_hits = sum(1 for term in _HISTORY_CORE_TERMS if _term_present(normalized_blob, term))
        history_noise_hits = sum(1 for term in _HISTORY_NOISE_TERMS if _term_present(normalized_blob, term))
        required_hits = max(1, min_hits)
        if history_hits < required_hits:
            return False
        if history_noise_hits >= 3 and history_hits <= 3:
            return False

    return True


def looks_like_technology_documentation(title: str, text: str, url: str) -> bool:
    normalized_title = normalize_text(title)
    normalized_blob = normalize_text(f"{title} {text}")
    url_low = (url or "").strip().lower()

    url_hits = sum(1 for hint in _TECH_DOC_URL_HINTS if hint in url_low)
    title_hits = sum(1 for term in _TECH_DOC_TITLE_HINTS if _term_present(normalized_title, term))
    text_hits = sum(1 for term in _TECH_DOC_TEXT_HINTS if _term_present(normalized_blob, term))
    noise_hits = sum(1 for term in _TECH_DOC_NOISE_TERMS if _term_present(normalized_blob, term))
    is_wikipedia = "wikipedia.org/wiki/" in url_low

    # For this collection we want implementation docs, not encyclopedia/reference pages.
    if is_wikipedia:
        return False

    # Avoid finance/news/captcha pages that can slip through search scraping.
    if noise_hits >= 3 and (url_hits + title_hits) == 0:
        return False
    if noise_hits >= 2 and text_hits <= 2:
        return False

    # Require at least one strong structural signal from URL or title.
    if (url_hits + title_hits) == 0:
        return False

    if (url_hits >= 1 or title_hits >= 1) and text_hits >= 2:
        return True
    if url_hits >= 2 and text_hits >= 1 and noise_hits == 0:
        return True

    return False


def looks_like_history_documentation(title: str, text: str, url: str) -> bool:
    normalized_title = normalize_text(title)
    normalized_blob = normalize_text(f"{title} {text}")
    url_low = (url or "").strip().lower()

    if "wikipedia.org/wiki/" in url_low:
        return False

    url_hits = sum(1 for hint in _HISTORY_DOC_URL_HINTS if hint in url_low)
    structure_title_hits = sum(1 for term in _HISTORY_DOC_STRUCTURE_TERMS if _term_present(normalized_title, term))
    structure_text_hits = sum(1 for term in _HISTORY_DOC_STRUCTURE_TERMS if _term_present(normalized_blob, term))
    history_hits = sum(1 for term in _HISTORY_CORE_TERMS if _term_present(normalized_blob, term))
    noise_hits = sum(1 for term in _HISTORY_NOISE_TERMS if _term_present(normalized_blob, term))

    if noise_hits >= 3 and url_hits == 0 and structure_title_hits == 0:
        return False

    if url_hits >= 1:
        return True
    if structure_title_hits >= 1 and (structure_text_hits >= 1 or history_hits >= 1):
        return True
    if history_hits >= 2:
        return True
    return structure_text_hits >= 3


def looks_like_access_interstitial(title: str, text: str, url: str = "") -> bool:
    normalized_title = normalize_text(title)
    normalized_text = normalize_text(text)
    url_low = (url or "").strip().lower()

    if any(hint in url_low for hint in _INTERSTITIAL_URL_HINTS):
        return True

    if any(_term_present(normalized_title, hint) for hint in _INTERSTITIAL_TITLE_HINTS):
        return True

    text_hits = sum(1 for hint in _INTERSTITIAL_TEXT_HINTS if _term_present(normalized_text, hint))
    if text_hits >= 2:
        # Challenge pages are typically short and repetitive.
        words = len(re.findall(r"\b\w+\b", text or ""))
        if words <= 500:
            return True

    return False


def _relevance_min_hits_for_collection(cfg: CollectionConfig) -> int:
    if cfg.source_type == "academic" and cfg.domain == "history":
        return 1
    if cfg.source_type == "technical" and cfg.domain == "history":
        return 1
    if cfg.source_type == "news" and cfg.domain == "history":
        return 1
    return 2


def clean_data(raw_docs: List[Dict], cfg: CollectionConfig) -> List[Dict]:
    cleaned: List[Dict] = []
    min_text_len = 120 if cfg.source_type in {"academic", "news"} else 200
    min_relevance_hits = _relevance_min_hits_for_collection(cfg)
    for doc in raw_docs:
        record = dict(doc)
        record["title"] = clean_text(record.get("title", ""))
        record["text"] = clean_text(record.get("text", ""))
        record = ensure_generated_metadata(
            record,
            source_name=str(record.get("source_name") or ""),
            default_domain=cfg.domain,
            default_source_type=cfg.source_type,
        )

        if looks_like_access_interstitial(
            title=record.get("title", ""),
            text=record.get("text", ""),
            url=record.get("url", ""),
        ):
            continue

        if len(record["text"]) < min_text_len:
            continue
        if not parse_date(record.get("publication_date", "")):
            continue
        if not is_english_text(f"{record['title']} {record['text']}"):
            continue
        if cfg.source_type == "news":
            if not is_news_doc_domain_relevant(
                title=record["title"],
                text=record["text"],
                domain=cfg.domain,
                keywords=cfg.keywords,
                min_hits=min_relevance_hits,
            ):
                continue
        else:
            if not looks_domain_relevant(
                f"{record['title']} {record['text']}",
                cfg.domain,
                cfg.keywords,
                min_hits=min_relevance_hits,
            ):
                continue

        if cfg.source_type == "technical" and cfg.domain == "technology":
            source_name_low = str(record.get("source_name") or "").strip().lower()
            # Keep this collection docs-only: reject encyclopedia fallback and enforce docs checks on web pages.
            if source_name_low == "wikipedia":
                continue
            if source_name_low in {"googlesearch", "bingsearch"}:
                if not looks_like_technology_documentation(
                    title=record.get("title", ""),
                    text=record.get("text", ""),
                    url=record.get("url", ""),
                ):
                    continue

        record["domain"] = cfg.domain
        record["source_type"] = cfg.source_type
        cleaned.append(record)
    return cleaned


def deduplicate_data(cleaned_docs: List[Dict], threshold: float = 0.95) -> Tuple[List[Dict], int]:
    if not cleaned_docs:
        return [], 0

    # Fast path for small batches: lightweight exact-ish key dedupe avoids model startup overhead.
    if len(cleaned_docs) <= 50:
        seen_keys = set()
        deduped: List[Dict] = []
        for doc in cleaned_docs:
            key = normalize_text(f"{doc.get('title', '')} {doc.get('text', '')[:500]}")
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(doc)
        return deduped, len(cleaned_docs) - len(deduped)

    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = get_embedding_model()

    texts = [normalize_text(f"{d.get('title', '')} {d.get('text', '')}") for d in cleaned_docs]
    embeddings = _EMBED_MODEL.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    kept_idx: List[int] = []
    removed = set()

    for i in range(len(cleaned_docs)):
        if i in removed:
            continue
        kept_idx.append(i)
        sims = embeddings @ embeddings[i]
        near = np.where(sims >= threshold)[0].tolist()
        for j in near:
            if j > i:
                removed.add(j)

    return [cleaned_docs[i] for i in kept_idx], len(removed)


def add_final_fields(docs: List[Dict], cfg: CollectionConfig) -> List[Dict]:
    out: List[Dict] = []
    for d in docs:
        record = ensure_generated_metadata(
            d,
            source_name=str(d.get("source_name") or ""),
            default_domain=cfg.domain,
            default_source_type=cfg.source_type,
        )
        score, label = compute_freshness(record.get("publication_date", ""), cfg.domain)
        record["freshness_score"] = score
        record["freshness_label"] = label
        out.append(record)
    return out
