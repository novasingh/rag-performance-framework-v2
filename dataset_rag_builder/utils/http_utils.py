from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import logging
import os
import random
import re
import threading
import time
from typing import Dict, Optional
from urllib.parse import urlparse

import requests


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        logging.warning("Invalid float for %s=%r; using default=%s", name, raw, default)
        return default


_REQUEST_COUNTER = 0
_REQUEST_COUNTER_LOCK = threading.Lock()
_THROTTLE_EVERY_N_REQUESTS = 10
_THROTTLE_MIN_SECONDS = 10.0
_THROTTLE_MAX_SECONDS = 15.0
_ENABLE_PERIODIC_THROTTLE = (os.getenv("RAG_ENABLE_PERIODIC_THROTTLE") or "").strip().lower() in {"1", "true", "yes", "on"}
_GDELT_HOST = "api.gdeltproject.org"
_WIKIPEDIA_HOST = "en.wikipedia.org"
_CROSSREF_HOST = "api.crossref.org"
_HOST_LAST_REQUEST_AT: Dict[str, float] = {}
_HOST_PACING_LOCK = threading.Lock()
_HOST_COOLDOWN_UNTIL: Dict[str, float] = {}
_HOST_COOLDOWN_LOCK = threading.Lock()
_HOST_MIN_INTERVAL_SECONDS: Dict[str, float] = {
    _GDELT_HOST: max(0.5, _env_float("RAG_GDELT_MIN_INTERVAL_SECONDS", 4.0)),
    _WIKIPEDIA_HOST: max(0.5, _env_float("RAG_WIKIPEDIA_MIN_INTERVAL_SECONDS", 1.4)),
    _CROSSREF_HOST: max(0.6, _env_float("RAG_CROSSREF_MIN_INTERVAL_SECONDS", 1.8)),
    "export.arxiv.org": 3.5,
}
_HTTP_429_BASE_WAIT_SECONDS = max(2.0, _env_float("RAG_HTTP_429_BASE_WAIT_SECONDS", 8.0))
_GDELT_429_BASE_WAIT_SECONDS = max(4.0, _env_float("RAG_GDELT_429_BASE_WAIT_SECONDS", 18.0))
_WIKIPEDIA_429_BASE_WAIT_SECONDS = max(4.0, _env_float("RAG_WIKIPEDIA_429_BASE_WAIT_SECONDS", 16.0))
_CROSSREF_429_BASE_WAIT_SECONDS = max(3.0, _env_float("RAG_CROSSREF_429_BASE_WAIT_SECONDS", 14.0))
_PROXY_POOL: Optional[list[str]] = None
_PROXY_INDEX = 0
_PROXY_LOCK = threading.Lock()
_PROXY_DISABLED_FOR_SESSION = False


def _base_wait_for_host(host: str) -> float:
    host_low = (host or "").strip().lower()
    if host_low == _GDELT_HOST:
        return _GDELT_429_BASE_WAIT_SECONDS
    if host_low == _CROSSREF_HOST:
        return _CROSSREF_429_BASE_WAIT_SECONDS
    if host_low == _WIKIPEDIA_HOST or host_low.endswith(".wikipedia.org"):
        return _WIKIPEDIA_429_BASE_WAIT_SECONDS
    return _HTTP_429_BASE_WAIT_SECONDS


def _apply_periodic_throttle(url: str) -> None:
    if not _ENABLE_PERIODIC_THROTTLE:
        return

    global _REQUEST_COUNTER
    with _REQUEST_COUNTER_LOCK:
        _REQUEST_COUNTER += 1
        current = _REQUEST_COUNTER

    if current % _THROTTLE_EVERY_N_REQUESTS == 0:
        sleep_for = random.uniform(_THROTTLE_MIN_SECONDS, _THROTTLE_MAX_SECONDS)
        logging.info("Throttle pause %.1fs at request #%s before %s", sleep_for, current, url)
        time.sleep(sleep_for)


def _apply_host_pacing(url: str) -> None:
    host = (urlparse(url).hostname or "").lower()
    min_interval = _HOST_MIN_INTERVAL_SECONDS.get(host)
    if not min_interval:
        return

    now = time.time()
    with _HOST_PACING_LOCK:
        last_ts = _HOST_LAST_REQUEST_AT.get(host)
        if last_ts is not None:
            elapsed = now - last_ts
            if elapsed < min_interval:
                sleep_for = min_interval - elapsed
                logging.info("Host pacing pause %.2fs for %s", sleep_for, host)
                time.sleep(sleep_for)
        _HOST_LAST_REQUEST_AT[host] = time.time()


def _apply_host_cooldown(url: str) -> None:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return

    now = time.time()
    with _HOST_COOLDOWN_LOCK:
        blocked_until = _HOST_COOLDOWN_UNTIL.get(host)

    if blocked_until and blocked_until > now:
        sleep_for = blocked_until - now
        logging.info("Host cooldown pause %.2fs for %s", sleep_for, host)
        time.sleep(sleep_for)


def _set_host_cooldown(host: str, wait_seconds: float) -> None:
    if wait_seconds <= 0:
        return

    now = time.time()
    new_until = now + wait_seconds
    with _HOST_COOLDOWN_LOCK:
        current_until = _HOST_COOLDOWN_UNTIL.get(host, 0.0)
        if new_until > current_until:
            _HOST_COOLDOWN_UNTIL[host] = new_until


def _parse_retry_after_seconds(header_value: str) -> float:
    value = (header_value or "").strip()
    if not value:
        return 0.0

    try:
        parsed = float(value)
        if parsed > 0:
            return parsed
    except Exception:
        pass

    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return 0.0


def _env_true(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _fetch_free_proxies() -> list[str]:
    # Public proxy lists are unstable and untrusted; keep this optional and bounded.
    if not _env_true("RAG_USE_FREE_PROXIES", default=False):
        return []

    endpoints = [
        "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all",
        "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    ]
    found: list[str] = []
    seen = set()

    for endpoint in endpoints:
        try:
            resp = requests.get(endpoint, timeout=20)
            if resp.status_code != 200:
                continue
            for line in resp.text.splitlines():
                val = line.strip()
                if not re.match(r"^\d{1,3}(?:\.\d{1,3}){3}:\d{2,5}$", val):
                    continue
                normalized = f"http://{val}"
                if normalized in seen:
                    continue
                seen.add(normalized)
                found.append(normalized)
                if len(found) >= 40:
                    break
        except Exception as err:
            logging.info("Free proxy source unavailable: %s (%s)", endpoint, err)
        if len(found) >= 40:
            break

    if found:
        logging.warning("Using %s public proxy endpoint(s) from free lists (untrusted).", len(found))
    return found


def _load_proxy_pool() -> list[str]:
    global _PROXY_POOL
    if _PROXY_POOL is not None:
        return _PROXY_POOL

    raw = (os.getenv("RAG_HTTP_PROXIES") or "").strip()
    if not raw:
        _PROXY_POOL = _fetch_free_proxies()
        return _PROXY_POOL

    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    normalized: list[str] = []
    for value in parts:
        if value.startswith("http://") or value.startswith("https://"):
            normalized.append(value)
        else:
            normalized.append(f"http://{value}")
    _PROXY_POOL = normalized
    logging.info("Loaded %s configured proxy endpoint(s)", len(_PROXY_POOL))
    return _PROXY_POOL


def _current_proxy() -> Optional[Dict[str, str]]:
    if _PROXY_DISABLED_FOR_SESSION:
        return None

    pool = _load_proxy_pool()
    if not pool:
        return None
    with _PROXY_LOCK:
        endpoint = pool[_PROXY_INDEX % len(pool)]
    return {"http": endpoint, "https": endpoint}


def _rotate_proxy(reason: str) -> None:
    global _PROXY_INDEX
    pool = _load_proxy_pool()
    if len(pool) <= 1:
        return
    with _PROXY_LOCK:
        _PROXY_INDEX = (_PROXY_INDEX + 1) % len(pool)
        endpoint = pool[_PROXY_INDEX]
    logging.warning("Rotated proxy after %s; new proxy=%s", reason, endpoint)


def _disable_proxies_for_session(reason: str) -> None:
    global _PROXY_DISABLED_FOR_SESSION
    if not _PROXY_DISABLED_FOR_SESSION:
        logging.warning("Disabling proxies for this run due to %s; using direct connection.", reason)
    _PROXY_DISABLED_FOR_SESSION = True


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    retries: int = 5,
    timeout: int = 45,
    min_delay: float = 1.0,
    max_delay: float = 2.0,
    retry_log_level: int = logging.WARNING,
    **kwargs,
) -> requests.Response:
    last_error: Optional[Exception] = None
    delay = random.uniform(min_delay, max_delay)
    force_direct = False
    host = (urlparse(url).hostname or "").lower()

    for attempt in range(1, retries + 1):
        try:
            _apply_host_cooldown(url)
            _apply_host_pacing(url)
            _apply_periodic_throttle(url)
            request_kwargs = dict(kwargs)
            if "proxies" not in request_kwargs and (not force_direct):
                # Use proxies on retries first to keep first attempt fast/stable.
                active_proxy = _current_proxy() if attempt > 1 else None
                if active_proxy:
                    request_kwargs["proxies"] = active_proxy
            response = session.request(method, url, timeout=timeout, **request_kwargs)
            if response.status_code in (429, 500, 502, 503, 504):
                response_text = (response.text or "")[:300].lower()
                rate_exceeded = "rate exceeded" in response_text
                status_code = response.status_code
                last_error = RuntimeError(f"HTTP {status_code} from {url}")
                logging.log(
                    retry_log_level,
                    "API status=%s for %s attempt=%s/%s%s",
                    status_code,
                    url,
                    attempt,
                    retries,
                    " (rate exceeded)" if rate_exceeded else "",
                )
                retry_after = 0.0
                if status_code == 429:
                    header_val = (response.headers.get("Retry-After") or "").strip()
                    retry_after = _parse_retry_after_seconds(header_val)
                    _rotate_proxy("HTTP 429")

                if attempt >= retries:
                    break

                if status_code == 429:
                    base_wait = _base_wait_for_host(host)
                    rate_limit_floor = max(base_wait, base_wait * (1.55 ** (attempt - 1)))
                    if host:
                        _set_host_cooldown(host, max(retry_after, rate_limit_floor))
                    sleep_for = max(delay, retry_after, rate_limit_floor, 10.0 if rate_exceeded else 0.0) + random.uniform(0.3, 1.2)
                else:
                    sleep_for = max(delay, 6.0) + random.uniform(0.2, 1.0)

                time.sleep(sleep_for)
                delay = min(delay * 1.7, 30.0)
                continue
            response.raise_for_status()
            return response
        except Exception as err:
            if isinstance(err, requests.HTTPError):
                status = err.response.status_code if err.response is not None else None
                if status is not None and 400 <= status < 500 and status != 429:
                    raise

            if isinstance(err, requests.exceptions.ProxyError):
                if not force_direct:
                    logging.warning("Proxy error for %s; switching to direct connection for remaining retries.", url)
                force_direct = True
                _rotate_proxy("proxy error")
                _disable_proxies_for_session("proxy connection errors")

            last_error = err
            logging.log(retry_log_level, "Request failed for %s attempt=%s/%s error=%s", url, attempt, retries, err)
            if attempt >= retries:
                break
            if isinstance(err, requests.Timeout):
                _rotate_proxy("timeout")
            time.sleep(delay)
            delay = min(delay * 1.7, 30.0)

    if last_error is None:
        last_error = RuntimeError("No response details")
    raise RuntimeError(f"Request failed after retries for {url}: {last_error}")


def response_json_safe(response: requests.Response) -> Dict:
    try:
        return response.json()
    except Exception:
        preview = (response.text or "")[:250]
        logging.warning("Non-JSON response ignored. Preview: %s", preview)
        return {}


def should_skip_article_url(url: str) -> bool:
    low = (url or "").strip().lower()
    if not low:
        return True
    blocked_markers = [
        "youtube.com",
        "youtu.be",
        "youtube-nocookie.com",
        "consent.yahoo.com",
        "guce.yahoo.com",
        "collectconsent",
        "/consent",
        "privacy-mgmt",
        "baijiahao.baidu.com",
    ]
    return any(marker in low for marker in blocked_markers)
