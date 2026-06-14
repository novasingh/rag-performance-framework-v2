from __future__ import annotations

import re
import warnings
from datetime import datetime, timezone
from typing import Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dateutil import parser as date_parser
from dateutil.parser import UnknownTimezoneWarning
from langdetect import LangDetectException, detect


_DATEUTIL_TZINFOS = {
    "UTC": 0,
    "UT": 0,
    "GMT": 0,
    "Z": 0,
    "EST": -5 * 3600,
    "EDT": -4 * 3600,
    "CST": -6 * 3600,
    "CDT": -5 * 3600,
    "MST": -7 * 3600,
    "MDT": -6 * 3600,
    "PST": -8 * 3600,
    "PDT": -7 * 3600,
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    parser_kind = "lxml"
    preview = (text or "").lstrip()[:240].lower()
    if preview.startswith("<?xml") or preview.startswith("<rss") or preview.startswith("<feed"):
        parser_kind = "xml"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(text, parser_kind)
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    plain = soup.get_text(separator=" ", strip=True)
    plain = re.sub(r"[^\w\s\.,;:!?\-\(\)/]", " ", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain


def parse_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        with warnings.catch_warnings():
            # Unknown timezone abbreviations can appear in RSS fields; prefer explicit
            # mapping and avoid warning spam for still-unknown tokens.
            warnings.filterwarnings("ignore", category=UnknownTimezoneWarning)
            dt = date_parser.parse(value, tzinfos=_DATEUTIL_TZINFOS)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def is_english_text(text: str) -> bool:
    sample = (text or "").strip()
    if len(sample) < 60:
        return False
    sample = sample[:2500]
    try:
        return detect(sample) == "en"
    except LangDetectException:
        ascii_letters = len(re.findall(r"[A-Za-z]", sample))
        ratio = ascii_letters / max(len(sample), 1)
        return ratio > 0.55
