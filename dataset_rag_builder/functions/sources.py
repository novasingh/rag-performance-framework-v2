from __future__ import annotations

import logging
import os
import re
import time
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from ..config import CollectionConfig
from ..functions.processing import build_doc, looks_like_access_interstitial, looks_like_history_documentation, looks_like_technology_documentation
from ..helpers.text_helpers import clean_text
from ..utils.http_utils import request_with_retry, response_json_safe, should_skip_article_url
from ..utils.io_utils import now_iso


MIN_SCRAPED_TEXT_CHARS = 120
_REDIRECT_QUERY_KEYS = ("url", "u", "q", "target", "dest")
_INTERSTITIAL_RETRY_WAIT_MS = 15000
_INTERSTITIAL_MAX_ATTEMPTS = 2
_HISTORY_SEED_PLAYWRIGHT_FALLBACK = (os.getenv("RAG_HISTORY_SEED_PLAYWRIGHT_FALLBACK") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        logging.warning("Invalid int for %s=%r; using default=%s", name, raw, default)
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    logging.warning("Invalid flag for %s=%r; using default=%s", name, raw, default)
    return default


_HISTORY_SEED_HOST_FAILURE_THRESHOLD = max(1, _env_int("RAG_HISTORY_SEED_HOST_FAILURE_THRESHOLD", 1))
_HISTORY_SEED_HOST_COOLDOWN_SECONDS = max(60, _env_int("RAG_HISTORY_SEED_HOST_COOLDOWN_SECONDS", 900))
_HISTORY_TECHNICAL_ALLOW_GOOGLE_SEARCH = _env_flag("RAG_HISTORY_TECHNICAL_ALLOW_GOOGLE_SEARCH", False)
_HISTORY_SEED_HOST_FAILURES: Dict[str, int] = {}
_HISTORY_SEED_HOST_COOLDOWN_UNTIL: Dict[str, float] = {}

_TECH_DOC_URL_HINTS = (
    "docs.",
    "/docs/",
    "/documentation/",
    "/learn/",
    "developer.",
    "/developer/",
    "/api/",
    "/reference/",
    "/guide/",
    "/tutorial",
    "/how-to/",
    "/manual/",
    "/sdk/",
    "readthedocs",
    "learn.microsoft.com",
    "developer.mozilla.org",
    "kubernetes.io/docs",
    "docs.aws.amazon.com",
    "cloud.google.com",
    "docs.docker.com",
    "developer.android.com",
    "go.dev/doc",
    "doc.rust-lang.org",
)

_TECH_DOC_URL_BLOCKLIST = (
    "wikipedia.org",
    "youtube.com",
    "baidu.com",
    "zhidao.baidu.com",
    "gamerant.com",
    "game8.co",
    "gamesradar.com",
    "screenrant.com",
    "esports.gg",
    "dict.leo.org",
    "stackoverflow.com/questions",
    "reddit.com",
    "medium.com",
    "quora.com",
)

_TECH_DOC_HOST_ALLOWLIST = (
    "kubernetes.io",
    "developer.mozilla.org",
    "learn.microsoft.com",
    "docs.aws.amazon.com",
    "cloud.google.com",
    "docs.python.org",
    "nodejs.org",
    "fastapi.tiangolo.com",
    "docs.djangoproject.com",
    "flask.palletsprojects.com",
    "docs.sqlalchemy.org",
    "docs.pydantic.dev",
    "react.dev",
    "nextjs.org",
    "vuejs.org",
    "angular.dev",
    "svelte.dev",
    "nuxt.com",
    "docs.docker.com",
    "docs.gitlab.com",
    "docs.github.com",
    "git-scm.com",
    "www.jenkins.io",
    "prometheus.io",
    "grafana.com",
    "www.elastic.co",
    "kafka.apache.org",
    "www.rabbitmq.com",
    "developer.hashicorp.com",
    "ansible.readthedocs.io",
    "nginx.org",
    "httpd.apache.org",
    "www.postgresql.org",
    "dev.mysql.com",
    "www.mongodb.com",
    "redis.io",
    "doc.rust-lang.org",
    "go.dev",
    "docs.oracle.com",
    "grpc.io",
    "spec.openapis.org",
    "developer.android.com",
)


_PUBMED_MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


def _pubmed_month_to_number(value: str) -> Optional[str]:
    raw = (value or "").strip()
    if not raw:
        return None
    match = re.search(r"\d{1,2}", raw)
    if match:
        month_num = int(match.group(0))
        if 1 <= month_num <= 12:
            return f"{month_num:02d}"
    key = re.sub(r"[^A-Za-z]", "", raw).lower()[:3]
    if not key:
        return None
    return _PUBMED_MONTHS.get(key)


def _pubmed_day_to_number(value: str) -> Optional[str]:
    raw = (value or "").strip()
    if not raw:
        return None
    match = re.search(r"\d{1,2}", raw)
    if not match:
        return None
    day_num = int(match.group(0))
    if 1 <= day_num <= 31:
        return f"{day_num:02d}"
    return None


def _format_pubmed_date(year: str, month: Optional[str], day: Optional[str]) -> str:
    year_value = (year or "").strip()
    if not year_value:
        return ""
    month_value = month or "01"
    day_value = day or "01"
    return f"{year_value}-{month_value}-{day_value}"


def _pubmed_date_from_node(article: ET.Element, base_path: str) -> str:
    year = article.findtext(f"{base_path}/Year", default="")
    month = _pubmed_month_to_number(article.findtext(f"{base_path}/Month", default=""))
    day = _pubmed_day_to_number(article.findtext(f"{base_path}/Day", default=""))
    return _format_pubmed_date(year, month, day)


def _pubmed_date_from_medline(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", raw) if t]
    year = ""
    month = None
    day = None
    for token in tokens:
        if re.fullmatch(r"(19|20)\d{2}", token):
            year = token
            break
    if not year:
        return ""

    year_index = tokens.index(year)
    for token in tokens[year_index + 1 :]:
        month = _pubmed_month_to_number(token)
        if month:
            break

    if month:
        for token in tokens[year_index + 1 :]:
            if _pubmed_month_to_number(token):
                continue
            day = _pubmed_day_to_number(token)
            if day:
                break
    else:
        numeric_tokens = [t for t in tokens[year_index + 1 :] if re.fullmatch(r"\d{1,2}", t)]
        if len(numeric_tokens) >= 1:
            month = _pubmed_month_to_number(numeric_tokens[0])
        if len(numeric_tokens) >= 2:
            day = _pubmed_day_to_number(numeric_tokens[1])

    return _format_pubmed_date(year, month, day)

_TECH_DOCUMENTATION_SEED_URLS = (
    "https://kubernetes.io/docs/concepts/overview/",
    "https://kubernetes.io/docs/reference/kubectl/",
    "https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands",
    "https://kubernetes.io/docs/tasks/",
    "https://kubernetes.io/docs/tutorials/",
    "https://kubernetes.io/docs/setup/",
    "https://kubernetes.io/docs/concepts/security/",
    "https://kubernetes.io/docs/concepts/services-networking/",
    "https://developer.mozilla.org/en-US/docs/Web/HTTP",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
    "https://developer.mozilla.org/en-US/docs/Web/API",
    "https://developer.mozilla.org/en-US/docs/Web/CSS",
    "https://developer.mozilla.org/en-US/docs/Web/HTML",
    "https://developer.mozilla.org/en-US/docs/Web/Performance",
    "https://developer.mozilla.org/en-US/docs/Web/Security",
    "https://developer.mozilla.org/en-US/docs/Web/Accessibility",
    "https://learn.microsoft.com/en-us/azure/architecture/",
    "https://learn.microsoft.com/en-us/azure/aks/",
    "https://learn.microsoft.com/en-us/azure/devops/",
    "https://learn.microsoft.com/en-us/azure/ai-services/",
    "https://learn.microsoft.com/en-us/azure/architecture/patterns/",
    "https://learn.microsoft.com/en-us/dotnet/core/introduction",
    "https://learn.microsoft.com/en-us/dotnet/csharp/",
    "https://learn.microsoft.com/en-us/aspnet/core/",
    "https://learn.microsoft.com/en-us/powershell/scripting/overview",
    "https://learn.microsoft.com/en-us/sql/sql-server/",
    "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
    "https://docs.aws.amazon.com/eks/latest/userguide/what-is-eks.html",
    "https://docs.aws.amazon.com/ec2/latest/userguide/concepts.html",
    "https://docs.aws.amazon.com/s3/latest/userguide/Welcome.html",
    "https://docs.aws.amazon.com/iam/latest/UserGuide/introduction.html",
    "https://docs.aws.amazon.com/cloudformation/index.html",
    "https://docs.aws.amazon.com/cloudwatch/index.html",
    "https://docs.aws.amazon.com/rds/latest/userguide/Welcome.html",
    "https://docs.aws.amazon.com/dynamodb/latest/developerguide/Introduction.html",
    "https://docs.aws.amazon.com/apigateway/latest/developerguide/welcome.html",
    "https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html",
    "https://cloud.google.com/kubernetes-engine/docs",
    "https://cloud.google.com/functions/docs",
    "https://cloud.google.com/run/docs",
    "https://cloud.google.com/compute/docs",
    "https://cloud.google.com/storage/docs",
    "https://cloud.google.com/sql/docs",
    "https://cloud.google.com/pubsub/docs",
    "https://cloud.google.com/bigquery/docs",
    "https://cloud.google.com/api-gateway/docs",
    "https://docs.python.org/3/tutorial/",
    "https://docs.python.org/3/library/index.html",
    "https://docs.python.org/3/reference/index.html",
    "https://docs.python.org/3/howto/index.html",
    "https://docs.python.org/3/faq/",
    "https://nodejs.org/docs/latest/api/",
    "https://nodejs.org/en/docs/guides/",
    "https://docs.djangoproject.com/en/stable/",
    "https://flask.palletsprojects.com/en/stable/",
    "https://fastapi.tiangolo.com/tutorial/",
    "https://docs.sqlalchemy.org/en/20/",
    "https://docs.pydantic.dev/latest/",
    "https://react.dev/learn",
    "https://react.dev/reference/react",
    "https://nextjs.org/docs",
    "https://vuejs.org/guide/introduction.html",
    "https://vuejs.org/api/",
    "https://angular.dev/overview",
    "https://angular.dev/guide",
    "https://svelte.dev/docs/svelte/overview",
    "https://nuxt.com/docs/getting-started/introduction",
    "https://docs.docker.com/get-started/",
    "https://docs.docker.com/engine/",
    "https://docs.docker.com/compose/",
    "https://docs.gitlab.com/ee/",
    "https://docs.github.com/en/actions",
    "https://docs.github.com/en/rest",
    "https://git-scm.com/doc",
    "https://www.jenkins.io/doc/",
    "https://prometheus.io/docs/introduction/overview/",
    "https://prometheus.io/docs/prometheus/latest/getting_started/",
    "https://grafana.com/docs/grafana/latest/",
    "https://www.elastic.co/guide/index.html",
    "https://kafka.apache.org/documentation/",
    "https://www.rabbitmq.com/docs",
    "https://developer.hashicorp.com/terraform/docs",
    "https://developer.hashicorp.com/vault/docs",
    "https://ansible.readthedocs.io/projects/ansible-core/devel/",
    "https://nginx.org/en/docs/",
    "https://httpd.apache.org/docs/",
    "https://www.postgresql.org/docs/current/",
    "https://dev.mysql.com/doc/",
    "https://www.mongodb.com/docs/",
    "https://redis.io/docs/latest/",
    "https://doc.rust-lang.org/book/",
    "https://doc.rust-lang.org/reference/",
    "https://go.dev/doc/",
    "https://go.dev/doc/tutorial/",
    "https://docs.oracle.com/en/java/javase/21/docs/api/",
    "https://grpc.io/docs/",
    "https://spec.openapis.org/oas/latest.html",
    "https://developer.android.com/guide",
    "https://pytorch.org/docs/stable/index.html",
    "https://pytorch.org/tutorials/",
    "https://www.tensorflow.org/guide",
    "https://www.tensorflow.org/api_docs",
)

_HISTORY_DOC_URL_HINTS = (
    "/archive",
    "/archives",
    "/collection",
    "/collections",
    "/catalog",
    "/manuscript",
    "/records",
    "/heritage",
    "/museum",
    "/chronicle",
    "/timeline",
    "/finding-aid",
    "digitalcollections",
    "chroniclingamerica",
    "iiif",
    "openarchives",
    "dublincore",
    "tei-c",
)

_HISTORY_DOC_URL_BLOCKLIST = (
    "wikipedia.org",
    "youtube.com",
    "reddit.com",
    "medium.com",
    "quora.com",
    "gamerant.com",
    "game8.co",
    "screenrant.com",
    "espn.com",
)

_HISTORY_DOC_HOST_ALLOWLIST = (
    "loc.gov",
    "chroniclingamerica.loc.gov",
    "id.loc.gov",
    "archives.gov",
    "nationalarchives.gov.uk",
    "europeana.eu",
    "pro.europeana.eu",
    "archive.org",
    "iiif.io",
    "ead3.archivists.org",
    "openarchives.org",
    "dublincore.org",
    "tei-c.org",
    "rightsstatements.org",
    "omeka.org",
    "bl.uk",
    "metmuseum.org",
    "britishmuseum.org",
    "vam.ac.uk",
    "si.edu",
    "getty.edu",
    "whc.unesco.org",
    "unesco.org",
    "digitalcollections.nypl.org",
    "digital.nls.uk",
    "rijksmuseum.nl",
)

_HISTORY_DOCUMENTATION_SEED_URLS = (
    "https://www.loc.gov/collections/",
    "https://www.loc.gov/manuscripts/",
    "https://www.loc.gov/preservation/",
    "https://www.loc.gov/standards/",
    "https://www.loc.gov/collections/world-digital-library/about-this-collection/",
    "https://chroniclingamerica.loc.gov/",
    "https://id.loc.gov/",
    "https://www.archives.gov/research",
    "https://www.archives.gov/research/catalog",
    "https://www.archives.gov/research/genealogy",
    "https://www.archives.gov/preservation",
    "https://www.archives.gov/open",
    "https://www.nationalarchives.gov.uk/help-with-your-research/",
    "https://www.nationalarchives.gov.uk/aboutapps/discovery/",
    "https://www.nationalarchives.gov.uk/information-management/",
    "https://www.nationalarchives.gov.uk/records-management/",
    "https://www.nationalarchives.gov.uk/archive-sector/",
    "https://www.europeana.eu/en/collections",
    "https://www.europeana.eu/en/search?query=manuscript",
    "https://pro.europeana.eu/page/iiif",
    "https://pro.europeana.eu/page/edm-documentation",
    "https://archive.org/details/texts",
    "https://archive.org/details/image",
    "https://archive.org/details/maps",
    "https://archive.org/developers/",
    "https://iiif.io/get-started/",
    "https://iiif.io/guides/",
    "https://iiif.io/api/image/3.0/",
    "https://iiif.io/api/presentation/3.0/",
    "https://iiif.io/api/search/1.0/",
    "https://ead3.archivists.org/",
    "https://www.openarchives.org/OAI/openarchivesprotocol.html",
    "https://www.dublincore.org/specifications/dublin-core/",
    "https://www.dublincore.org/specifications/dublin-core/dces/",
    "https://tei-c.org/guidelines/",
    "https://rightsstatements.org/page/1.0/",
    "https://omeka.org/classic/docs/",
    "https://omeka.org/s/docs/user-manual/",
    "https://www.bl.uk/collection-guides",
    "https://www.bl.uk/manuscripts",
    "https://www.bl.uk/research-guides",
    "https://www.metmuseum.org/art/collection",
    "https://www.metmuseum.org/about-the-met/collection-areas",
    "https://www.britishmuseum.org/collection",
    "https://www.britishmuseum.org/research/collection-online",
    "https://www.vam.ac.uk/collections",
    "https://www.vam.ac.uk/info/research",
    "https://www.si.edu/collections",
    "https://www.si.edu/openaccess",
    "https://www.getty.edu/research/collections/",
    "https://www.getty.edu/research/tools/",
    "https://whc.unesco.org/en/list/",
    "https://www.unesco.org/en/culture/heritage",
    "https://digitalcollections.nypl.org/",
    "https://digital.nls.uk/",
    "https://www.rijksmuseum.nl/en/collection",
)


def _seed_host_from_url(url: str) -> str:
    host = (urlparse(url or "").hostname or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _history_seed_host_is_blocked(host: str) -> bool:
    if not host:
        return False
    blocked_until = _HISTORY_SEED_HOST_COOLDOWN_UNTIL.get(host, 0.0)
    return blocked_until > time.time()


def _history_seed_host_note_success(host: str) -> None:
    if not host:
        return
    _HISTORY_SEED_HOST_FAILURES.pop(host, None)


def _history_seed_host_note_failure(host: str) -> None:
    if not host:
        return
    count = _HISTORY_SEED_HOST_FAILURES.get(host, 0) + 1
    _HISTORY_SEED_HOST_FAILURES[host] = count
    if count < _HISTORY_SEED_HOST_FAILURE_THRESHOLD:
        return

    cooldown_until = time.time() + _HISTORY_SEED_HOST_COOLDOWN_SECONDS
    prior = _HISTORY_SEED_HOST_COOLDOWN_UNTIL.get(host, 0.0)
    if cooldown_until > prior:
        _HISTORY_SEED_HOST_COOLDOWN_UNTIL[host] = cooldown_until
        logging.info(
            "history seed host cooldown host=%s failures=%s cooldown_seconds=%s",
            host,
            count,
            _HISTORY_SEED_HOST_COOLDOWN_SECONDS,
        )


def _safe_close_page(page: Optional[Any]) -> None:
    if page is None:
        return
    try:
        page.close()
    except Exception:
        pass


def _extract_seed_child_links(
    cfg: CollectionConfig,
    base_url: str,
    html: str,
    excluded: Set[str],
    max_links: int = 8,
) -> List[str]:
    if not html:
        return []

    links: List[str] = []
    seen: Set[str] = set()
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return []

    for anchor in soup.select("a[href]"):
        if len(links) >= max_links:
            break

        href = (anchor.get("href") or "").strip()
        if not href:
            continue

        candidate = urljoin(base_url, href).split("#", 1)[0]
        low = candidate.lower()
        if not candidate.startswith("http"):
            continue
        if low in excluded or low in seen:
            continue
        seen.add(low)
        if should_skip_article_url(candidate) or not _is_docs_candidate_for_collection(cfg, candidate):
            continue
        links.append(candidate)

    return links


def _build_seed_urls(seed_urls: Tuple[str, ...], seed_offset: int = 0) -> List[str]:
    urls: List[str] = []
    seen: Set[str] = set()
    for url in seed_urls:
        candidate = (url or "").strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        urls.append(candidate)

    if not urls:
        return urls
    if seed_offset <= 0:
        return urls

    offset = seed_offset % len(urls)
    return urls[offset:] + urls[:offset]


def _build_technology_seed_urls(seed_offset: int = 0) -> List[str]:
    return _build_seed_urls(_TECH_DOCUMENTATION_SEED_URLS, seed_offset=seed_offset)


def _build_history_seed_urls(seed_offset: int = 0) -> List[str]:
    return _build_seed_urls(_HISTORY_DOCUMENTATION_SEED_URLS, seed_offset=seed_offset)


def _is_docs_candidate_for_collection(cfg: CollectionConfig, url: str) -> bool:
    low = (url or "").strip().lower()
    if not low.startswith("http"):
        return False
    if cfg.source_type == "technical" and cfg.domain == "technology":
        if any(noise in low for noise in _TECH_DOC_URL_BLOCKLIST):
            return False
        host = (urlparse(low).hostname or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        host_allowed = any(host == allowed or host.endswith(f".{allowed}") for allowed in _TECH_DOC_HOST_ALLOWLIST)
        return host_allowed or any(hint in low for hint in _TECH_DOC_URL_HINTS)

    if cfg.source_type == "technical" and cfg.domain == "history":
        if any(noise in low for noise in _HISTORY_DOC_URL_BLOCKLIST):
            return False
        host = (urlparse(low).hostname or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        host_allowed = any(host == allowed or host.endswith(f".{allowed}") for allowed in _HISTORY_DOC_HOST_ALLOWLIST)
        return host_allowed or any(hint in low for hint in _HISTORY_DOC_URL_HINTS)

    return True


def collect_technology_documentation_seeds(
    session: requests.Session,
    browser: Optional[Any],
    cfg: CollectionConfig,
    max_results: int = 12,
    seed_offset: int = 0,
    exclude_urls: Optional[Set[str]] = None,
) -> List[Dict]:
    if cfg.source_type != "technical" or cfg.domain != "technology":
        return []

    excluded = {(u or "").strip().lower() for u in (exclude_urls or set()) if u}
    seed_urls = _build_technology_seed_urls(seed_offset=seed_offset)

    docs: List[Dict] = []
    for seed_url in seed_urls:
        if len(docs) >= max_results:
            break
        if should_skip_article_url(seed_url) or not _is_docs_candidate_for_collection(cfg, seed_url):
            continue

        seed_title = ""
        seed_text = ""
        seed_date = ""
        seed_final_url = seed_url
        seed_html = ""

        try:
            response = request_with_retry(session, "GET", seed_url, retries=1, timeout=12)
            seed_final_url = (response.url or seed_url or "").strip()
            seed_html = response.text or ""
            soup = BeautifulSoup(seed_html, "lxml")
            seed_title = soup.title.get_text(" ", strip=True) if soup.title else ""
            seed_text = clean_text(seed_html)
        except Exception:
            pass

        if (not seed_text) and browser is not None:
            seed_title, seed_text, seed_date, seed_final_url = fetch_article_text_playwright_with_url(
                browser,
                seed_url,
                post_load_wait_ms=800,
            )

        if seed_text:
            error_blob = f"{seed_title} {seed_text[:500]}".lower()
            resolved_url = (seed_final_url or seed_url or "").strip()
            resolved_low = resolved_url.lower()
            if not (("404" in error_blob and "not found" in error_blob) or "page not found" in error_blob):
                if (
                    resolved_low not in excluded
                    and not should_skip_article_url(resolved_url)
                    and _is_docs_candidate_for_collection(cfg, resolved_url)
                ):
                    if not looks_like_history_documentation(seed_title, seed_text, resolved_url):
                        continue
                    host = (urlparse(resolved_url).hostname or "").strip().lower()
                    if host.startswith("www."):
                        host = host[4:]
                    source_name = f"DocsSeed:{host}" if host else "DocsSeed"
                    docs.append(
                        build_doc(
                            cfg.domain,
                            cfg.source_type,
                            source_name,
                            seed_title,
                            seed_text,
                            seed_date or now_iso(),
                            resolved_url,
                            None,
                        )
                    )
                    excluded.add(resolved_low)
                    if len(docs) >= max_results:
                        break

        # Even when seed URL is already known, use it as a hub to discover fresh docs links.
        child_links = _extract_seed_child_links(
            cfg=cfg,
            base_url=seed_final_url or seed_url,
            html=seed_html,
            excluded=excluded,
            max_links=16,
        )

        for link in child_links:
            if len(docs) >= max_results:
                break

            title, text, date, final_url = fetch_article_text_requests_with_url(
                session,
                link,
                timeout=10,
                retries=1,
                interstitial_retry=False,
            )
            if (not text) and browser is not None:
                title, text, date, final_url = fetch_article_text_playwright_with_url(
                    browser,
                    link,
                    post_load_wait_ms=800,
                )
            if not title and not text:
                continue

            resolved_url = (final_url or link or "").strip()
            resolved_low = resolved_url.lower()
            if resolved_low in excluded:
                continue
            if should_skip_article_url(resolved_url) or not _is_docs_candidate_for_collection(cfg, resolved_url):
                continue
            if not looks_like_history_documentation(title, text, resolved_url):
                continue

            error_blob = f"{title} {text[:500]}".lower()
            if ("404" in error_blob and "not found" in error_blob) or "page not found" in error_blob:
                continue

            host = (urlparse(resolved_url).hostname or "").strip().lower()
            if host.startswith("www."):
                host = host[4:]
            source_name = f"DocsSeed:{host}" if host else "DocsSeed"
            docs.append(build_doc(cfg.domain, cfg.source_type, source_name, title, text, date or now_iso(), resolved_url, None))
            excluded.add(resolved_low)

    return docs


def collect_history_documentation_seeds(
    session: requests.Session,
    browser: Optional[Any],
    cfg: CollectionConfig,
    max_results: int = 12,
    seed_offset: int = 0,
    exclude_urls: Optional[Set[str]] = None,
) -> List[Dict]:
    if cfg.source_type != "technical" or cfg.domain != "history":
        return []

    excluded = {(u or "").strip().lower() for u in (exclude_urls or set()) if u}
    seed_urls = _build_history_seed_urls(seed_offset=seed_offset)

    docs: List[Dict] = []
    for seed_url in seed_urls:
        if len(docs) >= max_results:
            break
        seed_host = _seed_host_from_url(seed_url)
        if _history_seed_host_is_blocked(seed_host):
            continue
        if seed_url.strip().lower() in excluded:
            continue
        if should_skip_article_url(seed_url) or not _is_docs_candidate_for_collection(cfg, seed_url):
            continue

        title, text, date, final_url = fetch_article_text_requests_with_url(
            session,
            seed_url,
            timeout=12,
            retries=1,
            interstitial_retry=False,
        )
        if (not text) and browser is not None and _HISTORY_SEED_PLAYWRIGHT_FALLBACK:
            title, text, date, final_url = fetch_article_text_playwright_with_url(
                browser,
                seed_url,
                post_load_wait_ms=800,
            )
        if not title and not text:
            _history_seed_host_note_failure(seed_host)
            continue

        _history_seed_host_note_success(seed_host)

        error_blob = f"{title} {text[:500]}".lower()
        if ("404" in error_blob and "not found" in error_blob) or "page not found" in error_blob:
            continue

        resolved_url = (final_url or seed_url or "").strip()
        resolved_low = resolved_url.lower()
        if resolved_low in excluded:
            continue
        if should_skip_article_url(resolved_url) or not _is_docs_candidate_for_collection(cfg, resolved_url):
            continue

        if not date:
            date = now_iso()
        host = (urlparse(resolved_url).hostname or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        source_name = f"HistorySeed:{host}" if host else "HistorySeed"

        docs.append(build_doc(cfg.domain, cfg.source_type, source_name, title, text, date, resolved_url, None))
        excluded.add(resolved_low)

    return docs


def extract_redirect_target_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return ""

    try:
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower()
        if ("bing.com" not in host) and ("news.google.com" not in host):
            return value

        query_vals = parse_qs(parsed.query)
        for key in _REDIRECT_QUERY_KEYS:
            candidate_list = query_vals.get(key) or []
            if not candidate_list:
                continue
            candidate = (candidate_list[0] or "").strip()
            if candidate.startswith("http://") or candidate.startswith("https://"):
                return candidate
    except Exception:
        return value

    return value


def resolve_final_article_url(session: requests.Session, url: str, timeout: int = 20) -> str:
    candidate = extract_redirect_target_url(url)
    if should_skip_article_url(candidate):
        return candidate
    try:
        response = request_with_retry(
            session,
            "GET",
            candidate,
            retries=2,
            timeout=timeout,
            allow_redirects=True,
            stream=True,
        )
        final_url = (response.url or candidate or "").strip()
        try:
            response.close()
        except Exception:
            pass
        if final_url:
            return final_url
    except Exception:
        pass
    return candidate


def fetch_article_text_requests_with_url(
    session: requests.Session,
    url: str,
    timeout: int = 20,
    retries: int = 2,
    interstitial_retry: bool = True,
) -> Tuple[str, str, str, str]:
    resolved_url = resolve_final_article_url(session, url, timeout=timeout)
    if should_skip_article_url(resolved_url):
        return "", "", "", resolved_url

    scrape_start = time.perf_counter()
    latest_url = resolved_url
    try:
        for attempt in range(1, _INTERSTITIAL_MAX_ATTEMPTS + 1):
            resp = request_with_retry(session, "GET", latest_url, retries=retries, timeout=timeout)
            final_url = (resp.url or latest_url or "").strip()
            html = resp.text
            content_type = (resp.headers.get("content-type") or "").lower()
            parser_kind = "xml" if ("xml" in content_type or (html or "").lstrip().lower().startswith("<?xml")) else "lxml"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
                soup = BeautifulSoup(html, parser_kind)
            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            date_val = ""
            for selector in [
                ("meta", {"property": "article:published_time"}),
                ("meta", {"name": "pubdate"}),
                ("meta", {"name": "date"}),
                ("time", {}),
            ]:
                tag = soup.find(selector[0], selector[1])
                if tag:
                    date_val = (tag.get("content") or tag.get_text(" ", strip=True) or "").strip()
                    if date_val:
                        break

            text = clean_text(html)
            latest_url = final_url or latest_url
            scrape_ms = int((time.perf_counter() - scrape_start) * 1000)

            if looks_like_access_interstitial(title=title, text=text, url=latest_url):
                if interstitial_retry and attempt < _INTERSTITIAL_MAX_ATTEMPTS:
                    logging.info(
                        "Interstitial detected (requests) url=%s scrape_ms=%s wait_ms=%s attempt=%s/%s",
                        latest_url,
                        scrape_ms,
                        _INTERSTITIAL_RETRY_WAIT_MS,
                        attempt,
                        _INTERSTITIAL_MAX_ATTEMPTS,
                    )
                    time.sleep(_INTERSTITIAL_RETRY_WAIT_MS / 1000.0)
                    continue

                logging.info("Skipping interstitial page (requests) url=%s scrape_ms=%s", latest_url, scrape_ms)
                return "", "", "", latest_url

            if len(text) < MIN_SCRAPED_TEXT_CHARS:
                return "", "", "", latest_url

            return title, text, date_val, latest_url

        return "", "", "", latest_url
    except Exception:
        return "", "", "", latest_url


def fetch_article_text_requests(session: requests.Session, url: str) -> Tuple[str, str, str]:
    title, text, date_val, _ = fetch_article_text_requests_with_url(session, url)
    return title, text, date_val


def fetch_article_text_playwright_with_url(
    browser: Optional[Any],
    url: str,
    post_load_wait_ms: int = 0,
) -> Tuple[str, str, str, str]:
    initial_url = extract_redirect_target_url(url)
    if browser is None or should_skip_article_url(initial_url):
        return "", "", "", initial_url
    page = None
    scrape_start = time.perf_counter()
    try:
        page = browser.new_page()
        page.goto(initial_url, timeout=45000, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        if post_load_wait_ms > 0:
            page.wait_for_timeout(post_load_wait_ms)
        html = page.content()
        final_url = (page.url or initial_url or "").strip()
        soup = BeautifulSoup(html, "lxml")
        title = page.title() or ""
        date_val = ""
        for selector in [
            ("meta", {"property": "article:published_time"}),
            ("meta", {"name": "pubdate"}),
            ("meta", {"name": "date"}),
            ("time", {}),
        ]:
            tag = soup.find(selector[0], selector[1])
            if tag:
                date_val = (tag.get("content") or tag.get_text(" ", strip=True) or "").strip()
                if date_val:
                    break
        text = clean_text(html)

        if looks_like_access_interstitial(title=title, text=text, url=final_url):
            scrape_ms = int((time.perf_counter() - scrape_start) * 1000)
            logging.info(
                "Interstitial detected (playwright) url=%s scrape_ms=%s wait_ms=%s",
                final_url,
                scrape_ms,
                _INTERSTITIAL_RETRY_WAIT_MS,
            )
            page.wait_for_timeout(_INTERSTITIAL_RETRY_WAIT_MS)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass

            html = page.content()
            final_url = (page.url or initial_url or "").strip()
            soup = BeautifulSoup(html, "lxml")
            title = page.title() or ""
            date_val = ""
            for selector in [
                ("meta", {"property": "article:published_time"}),
                ("meta", {"name": "pubdate"}),
                ("meta", {"name": "date"}),
                ("time", {}),
            ]:
                tag = soup.find(selector[0], selector[1])
                if tag:
                    date_val = (tag.get("content") or tag.get_text(" ", strip=True) or "").strip()
                    if date_val:
                        break
            text = clean_text(html)

            if looks_like_access_interstitial(title=title, text=text, url=final_url):
                scrape_ms = int((time.perf_counter() - scrape_start) * 1000)
                logging.info("Skipping interstitial page (playwright) url=%s scrape_ms=%s", final_url, scrape_ms)
                return "", "", "", final_url or initial_url

        if len(text) < MIN_SCRAPED_TEXT_CHARS:
            return "", "", "", final_url or initial_url
        return title, text, date_val, final_url or initial_url
    except Exception:
        return "", "", "", initial_url
    finally:
        _safe_close_page(page)


def fetch_article_text_playwright(
    browser: Optional[Any],
    url: str,
    post_load_wait_ms: int = 0,
) -> Tuple[str, str, str]:
    title, text, date_val, _ = fetch_article_text_playwright_with_url(
        browser,
        url,
        post_load_wait_ms=post_load_wait_ms,
    )
    return title, text, date_val


def parse_arxiv(session: requests.Session, cfg: CollectionConfig, keyword: str, start: int, size: int) -> List[Dict]:
    response = request_with_retry(
        session,
        "GET",
        "https://export.arxiv.org/api/query",
        retries=6,
        timeout=60,
        min_delay=2.0,
        max_delay=5.0,
        params={
            "search_query": f"all:{keyword}",
            "start": start,
            "max_results": size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        },
    )
    root = ET.fromstring(response.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    docs: List[Dict] = []

    for entry in root.findall("a:entry", ns):
        title = entry.findtext("a:title", default="", namespaces=ns)
        abstract = entry.findtext("a:summary", default="", namespaces=ns)
        date = entry.findtext("a:published", default="", namespaces=ns)
        url = entry.findtext("a:id", default="", namespaces=ns)
        pdf_url = ""
        for link in entry.findall("a:link", ns):
            href = (link.attrib.get("href") or "").strip()
            link_type = (link.attrib.get("type") or "").strip()
            if ("/pdf/" in href) or (link_type == "application/pdf"):
                pdf_url = href
                break
        if (not pdf_url) and url:
            m = re.search(r"arxiv\.org/abs/([^/?#]+)", url)
            if m:
                pdf_url = f"https://arxiv.org/pdf/{m.group(1)}.pdf"

        authors = [a.findtext("a:name", default="", namespaces=ns) for a in entry.findall("a:author", ns)]
        doc = build_doc(cfg.domain, cfg.source_type, "arXiv", title, abstract, date, url, ", ".join([a for a in authors if a]) or None)
        if pdf_url:
            doc["pdf_url"] = pdf_url
        docs.append(doc)
    return docs


def parse_pubmed(
    session: requests.Session,
    cfg: CollectionConfig,
    keyword: str,
    start: int,
    size: int,
    mindate: str | None = None,
    maxdate: str | None = None,
    datetype: str = "pdat",
) -> List[Dict]:
    eutils = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    search_params = {
        "db": "pubmed",
        "term": keyword,
        "retmax": size,
        "retstart": start,
        "retmode": "json",
        "sort": "pub date",
    }
    if mindate:
        search_params["mindate"] = mindate
    if maxdate:
        search_params["maxdate"] = maxdate
    if mindate or maxdate:
        search_params["datetype"] = datetype

    search = request_with_retry(
        session,
        "GET",
        f"{eutils}/esearch.fcgi",
        params=search_params,
    )
    ids = response_json_safe(search).get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    fetch = request_with_retry(
        session,
        "GET",
        f"{eutils}/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract"},
    )
    root = ET.fromstring(fetch.text)
    docs: List[Dict] = []

    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle", default="")
        abstract_parts = [n.text or "" for n in article.findall(".//AbstractText")]
        abstract = " ".join([t.strip() for t in abstract_parts if t.strip()])
        pmid = article.findtext(".//PMID", default="")
        pub_date = _pubmed_date_from_node(article, ".//PubDate")
        article_date = _pubmed_date_from_node(article, ".//ArticleDate")
        medline = _pubmed_date_from_medline(article.findtext(".//PubDate/MedlineDate", default=""))
        date = pub_date or article_date or medline
        if not date:
            completed = _pubmed_date_from_node(article, ".//DateCompleted")
            created = _pubmed_date_from_node(article, ".//DateCreated")
            date = completed or created
        if not date:
            year = article.findtext(".//PubDate/Year", default="") or article.findtext(".//ArticleDate/Year", default="")
            date = _format_pubmed_date(year, None, None)
        authors = []
        for author in article.findall(".//Author"):
            fore = author.findtext("ForeName", default="")
            last = author.findtext("LastName", default="")
            name = f"{fore} {last}".strip()
            if name:
                authors.append(name)

        docs.append(
            build_doc(
                cfg.domain,
                cfg.source_type,
                "PubMed",
                title,
                abstract,
                date,
                f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                ", ".join(authors) if authors else None,
            )
        )
    return docs


def parse_wikipedia(session: requests.Session, cfg: CollectionConfig, keyword: str, offset: int, size: int) -> List[Dict]:
    api = "https://en.wikipedia.org/w/api.php"
    search = request_with_retry(
        session,
        "GET",
        api,
        retries=3,
        timeout=22,
        retry_log_level=logging.INFO,
        params={"action": "query", "list": "search", "format": "json", "srsearch": keyword, "srlimit": size, "sroffset": offset},
    )
    pages = response_json_safe(search).get("query", {}).get("search", [])
    if not pages:
        return []

    page_ids = [str(p.get("pageid")) for p in pages if p.get("pageid")]
    details = request_with_retry(
        session,
        "GET",
        api,
        retries=3,
        timeout=24,
        retry_log_level=logging.INFO,
        params={
            "action": "query",
            "format": "json",
            "pageids": "|".join(page_ids),
            "prop": "extracts|info|revisions",
            "explaintext": 1,
            "inprop": "url",
            "rvprop": "timestamp",
        },
    )
    details_json = response_json_safe(details)
    use_summary_fallback = not (cfg.source_type == "academic" and cfg.domain == "history")

    docs: List[Dict] = []
    for page in details_json.get("query", {}).get("pages", {}).values():
        title = page.get("title", "")
        text = page.get("extract", "")
        if (not text) and use_summary_fallback:
            summary = request_with_retry(
                session,
                "GET",
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}",
                retries=2,
                timeout=14,
                retry_log_level=logging.INFO,
            )
            text = (response_json_safe(summary).get("extract") or "").strip()

        revisions = page.get("revisions", [])
        date = revisions[0].get("timestamp", "") if revisions else ""
        if not date:
            date = "2000-01-01T00:00:00Z"

        docs.append(build_doc(cfg.domain, cfg.source_type, "Wikipedia", title, text, date, page.get("fullurl", ""), None))
    return docs


def parse_gdelt(session: requests.Session, cfg: CollectionConfig, keyword: str, timespan: str, max_records: int = 120) -> List[Dict]:
    query_tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", keyword) if len(t) >= 4]
    q = " ".join(query_tokens) if query_tokens else cfg.domain
    if cfg.domain not in q.lower():
        q = f"{q} {cfg.domain}"

    response = request_with_retry(
        session,
        "GET",
        "https://api.gdeltproject.org/api/v2/doc/doc",
        retries=5,
        timeout=35,
        min_delay=2.0,
        max_delay=4.0,
        retry_log_level=logging.INFO,
        params={
            "query": q,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": max_records,
            "sort": "HybridRel",
            "timespan": timespan,
        },
    )
    payload = response_json_safe(response)
    docs: List[Dict] = []

    for item in payload.get("articles", []):
        docs.append(
            build_doc(
                cfg.domain,
                cfg.source_type,
                item.get("domain", "GDELT"),
                item.get("title", ""),
                item.get("title", ""),
                item.get("seendate", ""),
                item.get("url", ""),
                item.get("sourcecountry", "") or None,
            )
        )
    return docs


def parse_bing_news_rss(session: requests.Session, cfg: CollectionConfig, keyword: str, size: int) -> List[Dict]:
    url = f"https://www.bing.com/news/search?q={quote(keyword)}&format=rss"
    response = request_with_retry(session, "GET", url, retries=2, timeout=18, retry_log_level=logging.INFO)
    root = ET.fromstring(response.text)
    docs: List[Dict] = []

    for item in root.findall("./channel/item")[: max(5, min(80, size))]:
        title = (item.findtext("title") or "").strip()
        link = extract_redirect_target_url((item.findtext("link") or "").strip())
        pub_date = (item.findtext("pubDate") or "").strip()
        source_name = (item.findtext("source") or "BingNews").strip()
        description = clean_text(item.findtext("description") or "")

        docs.append(
            build_doc(
                cfg.domain,
                cfg.source_type,
                source_name or "BingNews",
                title,
                description or title,
                pub_date,
                link,
                None,
            )
        )

    return docs


def parse_crossref(session: requests.Session, cfg: CollectionConfig, keyword: str, size: int) -> List[Dict]:
    response = request_with_retry(
        session,
        "GET",
        "https://api.crossref.org/works",
        retries=3,
        timeout=30,
        params={
            "query": keyword,
            "rows": max(5, min(60, size)),
            "sort": "published",
            "order": "desc",
            "select": "DOI,title,abstract,author,published-print,published-online,created,URL,container-title",
        },
    )
    payload = response_json_safe(response)
    items = payload.get("message", {}).get("items", [])
    docs: List[Dict] = []

    for item in items:
        title_list = item.get("title") or []
        title = title_list[0].strip() if title_list else ""
        abstract = clean_text(item.get("abstract") or "")

        date_val = ""
        for key in ["published-print", "published-online", "created"]:
            parts = (item.get(key) or {}).get("date-parts") or []
            if parts and parts[0]:
                ymd = parts[0]
                year = str(ymd[0]) if len(ymd) >= 1 else ""
                month = str(ymd[1]).zfill(2) if len(ymd) >= 2 else "01"
                day = str(ymd[2]).zfill(2) if len(ymd) >= 3 else "01"
                if year:
                    date_val = f"{year}-{month}-{day}"
                    break

        url = (item.get("URL") or "").strip()
        doi = (item.get("DOI") or "").strip()
        if not url and doi:
            url = f"https://doi.org/{doi}"

        authors = []
        for author in item.get("author") or []:
            given = (author.get("given") or "").strip()
            family = (author.get("family") or "").strip()
            full = f"{given} {family}".strip()
            if full:
                authors.append(full)

        source_name = "Crossref"
        container_titles = item.get("container-title") or []
        if container_titles and container_titles[0]:
            source_name = container_titles[0]

        docs.append(
            build_doc(
                cfg.domain,
                cfg.source_type,
                source_name,
                title,
                abstract or title,
                date_val,
                url,
                ", ".join(authors) if authors else None,
            )
        )

    return docs


def parse_google_news_rss(session: requests.Session, cfg: CollectionConfig, keyword: str, size: int) -> List[Dict]:
    url = f"https://news.google.com/rss/search?q={quote(keyword)}"
    response = request_with_retry(session, "GET", url, retries=2, timeout=18)
    root = ET.fromstring(response.text)
    docs: List[Dict] = []

    for item in root.findall("./channel/item")[: max(5, min(80, size))]:
        title = (item.findtext("title") or "").strip()
        link = extract_redirect_target_url((item.findtext("link") or "").strip())
        pub_date = (item.findtext("pubDate") or "").strip()
        source_tag = item.find("source")
        source_name = source_tag.text.strip() if source_tag is not None and source_tag.text else "GoogleNews"
        description = clean_text(item.findtext("description") or "")

        docs.append(
            build_doc(
                cfg.domain,
                cfg.source_type,
                source_name,
                title,
                description or title,
                pub_date,
                link,
                None,
            )
        )

    return docs


def scrape_bing_search(
    browser: Optional[Any],
    cfg: CollectionConfig,
    keyword: str,
    max_results: int,
    session: Optional[requests.Session] = None,
) -> List[Dict]:
    if browser is None and session is None:
        return []

    results: List[Dict] = []
    query = _search_query_for_collection(cfg, keyword)
    page = None
    try:
        if browser is not None:
            page = browser.new_page()
            page.goto(
                f"https://www.bing.com/search?{urlencode({'q': query, 'setlang': 'en'})}",
                timeout=15000,
                wait_until="domcontentloaded",
            )
            html = page.content()
        else:
            response = request_with_retry(
                session,
                "GET",
                "https://www.bing.com/search",
                retries=2,
                timeout=20,
                params={"q": query, "setlang": "en"},
            )
            html = response.text

        soup = BeautifulSoup(html, "lxml")
        links = []
        for a in soup.select("li.b_algo h2 a"):
            href = (a.get("href") or "").strip()
            if href.startswith("http"):
                links.append(href)

        if not links:
            for a in soup.select("a"):
                href = (a.get("href") or "").strip()
                low = href.lower()
                if not href.startswith("http"):
                    continue
                if "bing.com" in low or "microsoft.com" in low:
                    continue
                links.append(href)

        for link in links[:max_results]:
            if should_skip_article_url(link) or not _is_docs_candidate_for_collection(cfg, link):
                continue

            title = ""
            text = ""
            date = ""
            final_url = link

            if session is not None:
                title, text, date, final_url = fetch_article_text_requests_with_url(
                    session,
                    link,
                    timeout=12,
                    retries=1,
                    interstitial_retry=False,
                )

            if (not text) and browser is not None:
                title, text, date, final_url = fetch_article_text_playwright_with_url(
                    browser,
                    link,
                    post_load_wait_ms=1500,
                )

            if not title and not text:
                continue
            if not date:
                date = now_iso()
            resolved_link = (final_url or link or "").strip()
            if should_skip_article_url(resolved_link) or not _is_docs_candidate_for_collection(cfg, resolved_link):
                continue
            results.append(build_doc(cfg.domain, cfg.source_type, "BingSearch", title, text, date, resolved_link, None))
    except Exception as err:
        logging.info("Bing search scraping skipped for keyword=%s error=%s", keyword, err)
    finally:
        _safe_close_page(page)
    return results


def _build_search_query(*parts: str) -> str:
    tokens: List[str] = []
    seen = set()
    for part in parts:
        part = (part or "").strip()
        if not part:
            continue
        
        # Don't split parts containing quotes or parentheses to keep logical blocks intact.
        if '"' in part or '(' in part or ')' in part:
            low = part.lower()
            if low not in seen:
                seen.add(low)
                tokens.append(part)
        else:
            for token in re.split(r"\s+", part):
                if not token:
                    continue
                low = token.lower()
                if low in seen:
                    continue
                seen.add(low)
                tokens.append(token)
    return " ".join(tokens)


def _search_query_for_collection(cfg: CollectionConfig, keyword: str) -> str:
    if cfg.source_type == "technical" and cfg.domain == "technology":
        return _build_search_query(
            keyword,
            cfg.domain,
            "official documentation",
            "developer documentation",
            "developer guide",
            "api reference",
            "sdk",
            "quickstart",
            "tutorial",
            "-wikipedia",
            "-news",
            "-finance",
        )

    if cfg.source_type == "technical" and cfg.domain == "history":
        return _build_search_query(
            keyword,
            cfg.domain,
            '("archive catalog" OR "digital collection" OR manuscript OR "historical records" OR "museum collection" OR "primary source" OR "reference guide")',
            "-wikipedia",
            "-news",
            "-sports",
        )

    if cfg.source_type == "news" and cfg.domain == "history":
        return _build_search_query(
            keyword,
            cfg.domain,
            "heritage",
            "archive",
            "museum",
            "archaeology",
            "repatriation",
            "historical records",
            "policy update",
            "-sports",
            "-score",
            "-match",
            "-site:support.google.com",
            "-site:maps.google.com",
            "-site:youtube.com",
            "-site:netflix.com",
            "-site:rottentomatoes.com",
        )

    return _build_search_query(keyword, cfg.domain, cfg.source_type)


def _extract_google_result_url(href: str) -> str:
    if not href:
        return ""
    if href.startswith("/url?"):
        parsed = urlparse(href)
        qv = parse_qs(parsed.query).get("q")
        if qv:
            return qv[0]
    if href.startswith("http"):
        return href
    return ""


def scrape_google_search(
    browser: Optional[Any],
    cfg: CollectionConfig,
    keyword: str,
    max_results: int,
    session: Optional[requests.Session] = None,
) -> List[Dict]:
    if browser is None and session is None:
        return []

    results: List[Dict] = []
    query = _search_query_for_collection(cfg, keyword)
    page = None
    try:
        if browser is not None:
            page = browser.new_page()
            page.goto(
                f"https://www.google.com/search?{urlencode({'q': query, 'hl': 'en'})}",
                timeout=15000,
                wait_until="domcontentloaded",
            )
            html = page.content()
        else:
            response = request_with_retry(
                session,
                "GET",
                "https://www.google.com/search",
                retries=2,
                timeout=20,
                params={"q": query, "hl": "en"},
            )
            html = response.text

        soup = BeautifulSoup(html, "lxml")
        links: List[str] = []
        for a in soup.select("a"):
            href = (a.get("href") or "").strip()
            candidate = _extract_google_result_url(href)
            if not candidate or should_skip_article_url(candidate):
                continue
            if "google.com" in candidate.lower() or "googleusercontent.com" in candidate.lower():
                continue
            if not _is_docs_candidate_for_collection(cfg, candidate):
                continue
            links.append(candidate)

        seen = set()
        for link in links:
            if len(results) >= max_results:
                break
            low = link.lower()
            if low in seen:
                continue
            seen.add(low)

            title = ""
            text = ""
            date = ""
            final_url = link

            if session is not None:
                title, text, date, final_url = fetch_article_text_requests_with_url(
                    session,
                    link,
                    timeout=12,
                    retries=1,
                    interstitial_retry=False,
                )

            if (not text) and browser is not None:
                title, text, date, final_url = fetch_article_text_playwright_with_url(
                    browser,
                    link,
                    post_load_wait_ms=1500,
                )

            if not title and not text:
                continue
            if not date:
                date = now_iso()
            resolved_link = (final_url or link or "").strip()
            if should_skip_article_url(resolved_link) or not _is_docs_candidate_for_collection(cfg, resolved_link):
                continue
            results.append(build_doc(cfg.domain, cfg.source_type, "GoogleSearch", title, text, date, resolved_link, None))
    except Exception as err:
        logging.warning("Google search scraping failed for keyword=%s error=%s", keyword, err)
    finally:
        _safe_close_page(page)

    return results


def scrape_web_search(
    browser: Optional[Any],
    cfg: CollectionConfig,
    keyword: str,
    max_results: int,
    session: Optional[requests.Session] = None,
) -> List[Dict]:
    # For history_technical we stay Bing-first; Google fallback is opt-in because it tends to 429.
    google_results: List[Dict] = []
    bing_results: List[Dict] = []
    
    if cfg.name == "history_technical":
        bing_results = scrape_bing_search(browser, cfg, keyword, max_results=max_results, session=session)
        remaining = max(0, max_results - len(bing_results))
        if remaining > 0 and _HISTORY_TECHNICAL_ALLOW_GOOGLE_SEARCH:
            google_results = scrape_google_search(browser, cfg, keyword, max_results=remaining, session=session)
    else:
        google_results = scrape_google_search(browser, cfg, keyword, max_results=max_results, session=session)
        remaining = max(0, max_results - len(google_results))
        if remaining > 0:
            bing_results = scrape_bing_search(browser, cfg, keyword, max_results=remaining, session=session)

    merged: List[Dict] = []
    seen_keys = set()
    ordered_results = bing_results + google_results if cfg.name == "history_technical" else google_results + bing_results
    for row in ordered_results:
        url_key = (row.get("url") or "").strip().lower()
        key = url_key or f"{(row.get('title') or '').strip().lower()}::{(row.get('source_name') or '').strip().lower()}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(row)
        if len(merged) >= max_results:
            break

    return merged


def download_arxiv_pdfs(
    session: requests.Session,
    base_dir: Path,
    cfg: CollectionConfig,
    docs: List[Dict],
    max_downloads: Optional[int] = None,
) -> None:
    if cfg.primary_source != "arxiv":
        return

    pdf_dir = base_dir / "raw" / "pdfs" / cfg.name
    pdf_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for doc in docs:
        if max_downloads is not None and downloaded >= max_downloads:
            break
        pdf_url = (doc.get("pdf_url") or "").strip()

        if not pdf_url:
            url = (doc.get("url") or "").strip()
            m = re.search(r"arxiv\.org/abs/([^/?#]+)", url)
            if m:
                pdf_url = f"https://arxiv.org/pdf/{m.group(1)}.pdf"
                doc["pdf_url"] = pdf_url
            else:
                continue

        arxiv_id = ""
        m = re.search(r"arxiv\.org/(?:pdf|abs)/([^/?#]+)", pdf_url)
        if m:
            arxiv_id = m.group(1).replace(".pdf", "")
        if not arxiv_id:
            url = (doc.get("url") or "").strip()
            m = re.search(r"arxiv\.org/abs/([^/?#]+)", url)
            if m:
                arxiv_id = m.group(1)

        file_stem = re.sub(r"[^A-Za-z0-9._-]", "_", arxiv_id or doc.get("id", "arxiv_doc"))
        pdf_path = pdf_dir / f"{file_stem}.pdf"

        doc["pdf_file_name"] = pdf_path.name
        doc["pdf_file_path"] = f"raw/pdfs/{cfg.name}/{pdf_path.name}"

        if pdf_path.exists():
            doc["pdf_available"] = True
            continue

        try:
            resp = request_with_retry(session, "GET", pdf_url, retries=3, timeout=60)
            pdf_path.write_bytes(resp.content)
            downloaded += 1
            doc["pdf_available"] = True
            logging.info("Downloaded arXiv PDF: %s", pdf_path)
        except Exception as err:
            doc["pdf_available"] = False
            logging.warning("Failed to download arXiv PDF for %s: %s", pdf_url, err)
