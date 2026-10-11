"""Tolerant JSON extraction for LLM replies (markdown fences, prose before/after, truncation)."""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json(raw: str):
    """Return the first JSON value found in `raw`. Raises ValueError if none parses."""
    text = _FENCE.sub("", raw.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in LLM reply: {raw[:300]!r}")
