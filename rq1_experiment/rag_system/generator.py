"""
rq1_experiment/rag_system/generator.py
=======================================
Dual-backend LLM generator: Ollama (local) or Google AI Studio.

Benchmark results (2026-05-13):
  ollama/llama3.1:8b          11.4s avg | 3/3 success | ~7 tok/s | no rate limit
  google/gemma-4-26b-a4b-it   13.6s avg | 2/2 success | rate-limited to 29 RPM

Active backend is controlled by GENERATOR_BACKEND in config.py.
Default: Ollama (faster, local, no quota).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from ..config import (
    GEMMA_MAX_OUTPUT_TOKENS,
    GEMMA_MIN_INTERVAL_SEC,
    GEMMA_MODEL_NAME,
    GEMMA_RPM_LIMIT,
    GEMMA_TEMPERATURE,
    GENERATOR_BACKEND,
    GOOGLE_AI_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RAG_PROMPT_TEMPLATE,
    DO_API_KEY,
    DO_MODEL_NAME,
    DO_CHAT_URL,
    DO_INPUT_PRICE_PER_M,
    DO_OUTPUT_PRICE_PER_M,
)

logger = logging.getLogger(__name__)

# ── API Usage Tracker (Singleton) ──────────────────────────────────────────────
import atexit

class TokenTracker:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TokenTracker, cls).__new__(cls)
            cls._instance.input_tokens = 0
            cls._instance.output_tokens = 0
            cls._instance.total_requests = 0
            cls._instance.start_time = time.time()
            atexit.register(cls._instance.print_summary)
        return cls._instance

    def add(self, input_tokens: int, output_tokens: int, requests: int = 1):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_requests += requests

    def print_summary(self):
        if self.input_tokens == 0 and self.output_tokens == 0:
            return
            
        elapsed = time.time() - self.start_time
        in_cost = (self.input_tokens / 1_000_000) * DO_INPUT_PRICE_PER_M
        out_cost = (self.output_tokens / 1_000_000) * DO_OUTPUT_PRICE_PER_M
        total_cost = in_cost + out_cost
        
        print("\n" + "="*50)
        print("API USAGE SUMMARY (DigitalOcean Inference)")
        print(f"Time Elapsed:   {elapsed/60:.2f} minutes")
        print(f"Total Requests: {self.total_requests:,}")
        print(f"Input Tokens:   {self.input_tokens:,} (${in_cost:.6f})")
        print(f"Output Tokens:  {self.output_tokens:,} (${out_cost:.6f})")
        print(f"Total Cost:     ${total_cost:.6f}")
        print("="*50 + "\n")

tracker = TokenTracker()

# ── Google AI rate-limit state ─────────────────────────────────────────────────
_rate_lock      = threading.Lock()
_last_call_time = 0.0
_call_count_min = 0
_window_start   = 0.0


def _rate_limited_sleep() -> None:
    """Enforce 29 RPM limit for Google AI backend."""
    global _last_call_time, _call_count_min, _window_start
    with _rate_lock:
        now = time.monotonic()
        if now - _window_start >= 60.0:
            _window_start   = now
            _call_count_min = 0
        if _call_count_min >= GEMMA_RPM_LIMIT:
            wait = 60.0 - (now - _window_start) + 0.5
            if wait > 0:
                logger.debug("RPM limit reached; sleeping %.2fs", wait)
                time.sleep(wait)
            now             = time.monotonic()
            _window_start   = now
            _call_count_min = 0
        elapsed = now - _last_call_time
        if elapsed < GEMMA_MIN_INTERVAL_SEC:
            time.sleep(GEMMA_MIN_INTERVAL_SEC - elapsed)
        _last_call_time  = time.monotonic()
        _call_count_min += 1


# ── DigitalOcean rate-limit state ──────────────────────────────────────────────
_do_rate_lock      = threading.Lock()
_do_minute_start   = 0.0
_do_hour_start     = 0.0
_do_count_min      = 0
_do_count_hour     = 0
# Track remaining quota from API response headers for proactive throttling
_do_remaining_requests = None   # last known x-ratelimit-remaining-requests
_do_limit_requests     = None   # last known x-ratelimit-limit-requests

def _update_do_rate_headers(resp_headers: dict) -> None:
    """Parse rate-limit headers from a DO API response and update global state."""
    global _do_remaining_requests, _do_limit_requests
    with _do_rate_lock:
        for key, value in resp_headers.items():
            kl = key.lower()
            if kl == "x-ratelimit-remaining-requests":
                try:
                    _do_remaining_requests = int(value)
                except (ValueError, TypeError):
                    pass
            elif kl == "x-ratelimit-limit-requests":
                try:
                    _do_limit_requests = int(value)
                except (ValueError, TypeError):
                    pass

def _do_rate_limited_sleep() -> None:
    """Enforce RPM and RPH limits for DigitalOcean backend with proactive throttling."""
    global _do_minute_start, _do_hour_start, _do_count_min, _do_count_hour
    from ..config import DO_RPM_LIMIT, DO_RPH_LIMIT
    
    with _do_rate_lock:
        now = time.monotonic()
        
        # Reset windows
        if now - _do_minute_start >= 60.0:
            _do_minute_start = now
            _do_count_min = 0
            
        if now - _do_hour_start >= 3600.0:
            _do_hour_start = now
            _do_count_hour = 0
            
        # Check hour limit
        if _do_count_hour >= DO_RPH_LIMIT:
            wait = 3600.0 - (now - _do_hour_start) + 1.0
            if wait > 0:
                logger.warning("DigitalOcean RPH limit (%d) reached; sleeping %.2fs", DO_RPH_LIMIT, wait)
                time.sleep(wait)
            now = time.monotonic()
            _do_hour_start = now
            _do_count_hour = 0
            # Also reset minute window since we slept past it
            _do_minute_start = now
            _do_count_min = 0
            
        # Check minute limit
        if _do_count_min >= DO_RPM_LIMIT:
            wait = 60.0 - (now - _do_minute_start) + 0.5
            if wait > 0:
                logger.debug("DigitalOcean RPM limit (%d) reached; sleeping %.2fs", DO_RPM_LIMIT, wait)
                time.sleep(wait)
            now = time.monotonic()
            _do_minute_start = now
            _do_count_min = 0

        # Proactive throttling: if remaining requests from API headers is low,
        # add a small delay to avoid bursting into a 429
        if _do_remaining_requests is not None and _do_remaining_requests <= 5:
            # Slow down when close to the limit
            throttle_wait = 2.0
            logger.debug("Proactive throttle: only %d requests remaining, sleeping %.1fs",
                         _do_remaining_requests, throttle_wait)
            time.sleep(throttle_wait)

        _do_count_min += 1
        _do_count_hour += 1


# ─────────────────────────────────────────────────────────────────────────────
# Ollama backend
# ─────────────────────────────────────────────────────────────────────────────

class OllamaBackend:
    """Local Ollama inference — no rate limits, fast, deterministic at temp=0."""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = GEMMA_TEMPERATURE,
        max_tokens: int = GEMMA_MAX_OUTPUT_TOKENS,
        timeout: int = 180,
        max_retries: int = 3,
    ) -> None:
        import requests
        self._requests = requests
        self.model       = model
        self.base_url    = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout
        self.max_retries = max_retries
        logger.info("OllamaBackend initialized: model=%s @ %s", model, base_url)

    def call(self, prompt: str) -> str:
        url     = f"{self.base_url}/api/generate"
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        attempt = 0
        backoff = 2.0
        while attempt <= self.max_retries:
            try:
                resp = self._requests.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    return (resp.json().get("response") or "").strip()
                else:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            except Exception as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                logger.warning("Ollama error (attempt %d): %s — retrying in %.1fs", attempt, exc, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
        return ""

    def test(self) -> bool:
        try:
            text = self.call("Reply with one word: hello")
            ok = len(text) > 0
            if ok:
                logger.info("Ollama test PASSED: %s", text[:60])
            return ok
        except Exception as exc:
            logger.error("Ollama test FAILED: %s", exc)
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Google AI backend
# ─────────────────────────────────────────────────────────────────────────────

class GoogleAIBackend:
    """Google AI Studio backend with 29 RPM rate limiting."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMMA_MODEL_NAME,
        temperature: float = GEMMA_TEMPERATURE,
        max_tokens: int = GEMMA_MAX_OUTPUT_TOKENS,
        max_retries: int = 5,
    ) -> None:
        from google import genai
        from google.genai import types as genai_types
        key = api_key or GOOGLE_AI_API_KEY
        if not key:
            raise ValueError("GOOGLE_AI_API_KEY not set.")
        self.client  = genai.Client(api_key=key)
        self.model   = model
        self.config  = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            candidate_count=1,
        )
        self.max_retries = max_retries
        logger.info("GoogleAIBackend initialized: model=%s, rpm=%d", model, GEMMA_RPM_LIMIT)

    def call(self, prompt: str) -> str:
        _rate_limited_sleep()
        attempt = 0
        backoff = 2.0
        while attempt <= self.max_retries:
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=self.config
                )
                return (resp.text or "").strip()
            except Exception as exc:
                err = str(exc).lower()
                attempt += 1
                if attempt > self.max_retries:
                    raise
                sleep = max(backoff, 30.0) if ("429" in err or "quota" in err or "rate" in err) else backoff
                logger.warning("Google API error (attempt %d): %s — retrying in %.1fs", attempt, exc, sleep)
                time.sleep(sleep)
                backoff = min(backoff * 2, 120.0)
        return ""

    def test(self) -> bool:
        try:
            text = self.call("Reply with one word: hello")
            ok = len(text) > 0
            if ok:
                logger.info("Google AI test PASSED: %s", text[:60])
            return ok
        except Exception as exc:
            logger.error("Google AI test FAILED: %s", exc)
            return False


# ─────────────────────────────────────────────────────────────────────────────
# DigitalOcean Inference backend
# ─────────────────────────────────────────────────────────────────────────────

class DigitalOceanBackend:
    """DigitalOcean Inference API (e.g., llama-4-maverick, gemma-4-31B-it)."""

    def __init__(
        self,
        api_key: str = DO_API_KEY,
        model: str = DO_MODEL_NAME,
        temperature: float = GEMMA_TEMPERATURE,
        max_tokens: int = GEMMA_MAX_OUTPUT_TOKENS,
        timeout: int = 180,
        max_retries: int = 5,
    ) -> None:
        import requests
        self._requests = requests
        self.api_key     = api_key
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout
        self.max_retries = max_retries
        self.url         = DO_CHAT_URL
        self.headers     = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        logger.info("DigitalOceanBackend initialized: model=%s", model)

    def call(self, prompt: str) -> str:
        import random
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        attempt = 0
        backoff = 2.0
        while attempt <= self.max_retries:
            _do_rate_limited_sleep()
            try:
                resp = self._requests.post(self.url, headers=self.headers, json=payload, timeout=self.timeout)

                # Always update rate-limit tracking from response headers
                _update_do_rate_headers(dict(resp.headers))

                if resp.status_code == 200:
                    result = resp.json()
                    
                    # Track token usage
                    usage = result.get("usage", {})
                    tracker.add(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                    
                    message_content = result.get("choices", [{}])[0].get("message", {}).get("content")
                    return message_content.strip() if message_content else ""

                elif resp.status_code == 429:
                    attempt += 1
                    if attempt > self.max_retries:
                        raise RuntimeError(
                            f"DigitalOcean 429 rate limit hit {self.max_retries} times "
                            f"for model {self.model}. Giving up on this request."
                        )

                    # Determine wait time: prefer Retry-After header, else exponential backoff + jitter
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait = float(retry_after) + random.uniform(0.5, 2.0)
                        except (ValueError, TypeError):
                            wait = backoff + random.uniform(0.5, 3.0)
                    else:
                        # Exponential backoff with jitter: 2s, 4s, 8s, 16s, 32s …
                        wait = backoff + random.uniform(0.5, 3.0)

                    remaining = resp.headers.get("x-ratelimit-remaining-requests", "?")
                    limit = resp.headers.get("x-ratelimit-limit-requests", "?")
                    logger.warning(
                        "DigitalOcean 429 (attempt %d/%d) model=%s "
                        "limit=%s remaining=%s — retrying in %.1fs",
                        attempt, self.max_retries, self.model,
                        limit, remaining, wait
                    )
                    time.sleep(wait)
                    backoff = min(backoff * 2, 120.0)
                    continue

                else:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            except Exception as exc:
                if isinstance(exc, SystemExit):
                    raise
                attempt += 1
                if attempt > self.max_retries:
                    raise
                wait = backoff + random.uniform(0.5, 2.0)
                logger.warning("DigitalOcean API error (attempt %d): %s — retrying in %.1fs", attempt, exc, wait)
                time.sleep(wait)
                backoff = min(backoff * 2, 60.0)
        return ""

    def test(self) -> bool:
        try:
            text = self.call("Reply with one word: hello")
            ok = len(text) > 0
            if ok:
                logger.info("DigitalOcean API test PASSED: %s", text[:60])
            return ok
        except Exception as exc:
            logger.error("DigitalOcean API test FAILED: %s", exc)
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Unified GemmaGenerator (backend-agnostic public interface)
# ─────────────────────────────────────────────────────────────────────────────

class GemmaGenerator:
    """
    Backend-agnostic LLM generator.

    Transparently wraps either OllamaBackend or GoogleAIBackend.
    The active backend is determined by config.GENERATOR_BACKEND.
    Public interface is identical regardless of backend.
    """

    def __init__(self, backend: Optional[str] = None) -> None:
        active = (backend or GENERATOR_BACKEND).lower()

        if active == "ollama":
            self._backend = OllamaBackend()
            self.backend_name = f"ollama/{OLLAMA_MODEL}"
        elif active == "google":
            self._backend = GoogleAIBackend()
            self.backend_name = f"google/{GEMMA_MODEL_NAME}"
        elif active == "digitalocean":
            self._backend = DigitalOceanBackend()
            self.backend_name = f"digitalocean/{DO_MODEL_NAME}"
        else:
            raise ValueError(f"Unknown backend: '{active}'. Use 'digitalocean', 'ollama', or 'google'.")

        logger.info("GemmaGenerator using backend: %s", self.backend_name)

    @staticmethod
    def _build_context(docs: List[Dict[str, Any]], max_chars_per_doc: int = 1200) -> str:
        parts: List[str] = []
        for i, doc in enumerate(docs, 1):
            title  = (doc.get("title")       or "").strip()
            text   = (doc.get("text")        or "").strip()[:max_chars_per_doc]
            source = (doc.get("source_name") or doc.get("source_type") or "").strip()
            date   = (doc.get("publication_date") or "").strip()
            header = f"[{i}]"
            if title:  header += f" {title}"
            if source:
                header += f" ({source}"
                if date: header += f", {date}"
                header += ")"
            parts.append(f"{header}\n{text}")
        return "\n\n".join(parts)

    def generate(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        max_context_chars: int = 1200,
    ) -> Dict[str, Any]:
        """
        Generate a grounded answer. Returns dict with: response, prompt, latency_ms, error.
        """
        context = self._build_context(retrieved_docs, max_chars_per_doc=max_context_chars)
        prompt  = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        try:
            t0         = time.perf_counter()
            text       = self._backend.call(prompt)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {"response": text, "prompt": prompt, "latency_ms": round(latency_ms, 2), "error": None}
        except Exception as exc:
            logger.error("Generator failed: %s", exc)
            return {"response": "", "prompt": prompt, "latency_ms": 0.0, "error": str(exc)}

    def generate_raw(self, prompt: str) -> str:
        """Direct prompt call (used by query_generator). Returns raw text."""
        try:
            return self._backend.call(prompt)
        except Exception as exc:
            logger.error("generate_raw failed: %s", exc)
            return ""

    def test_connection(self) -> bool:
        return self._backend.test()
