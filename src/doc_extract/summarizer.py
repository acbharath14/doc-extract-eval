"""LLM summarization (Ollama-first) plus deterministic helpers.

- summarize_ollama: ask a local Ollama model for a business summary
- extract_entities: regex dates, currency amounts, proper names
- classify_document: heuristic invoice | contract | report | letter | unknown
"""
from __future__ import annotations

import json
import os
import re
import urllib.request


# --- Entity extraction (deterministic, no LLM) ---

_DATE_RES = [
    # 2026-10-11, 10/11/2026, 11-10-2026
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    # October 11, 2026 / 11 October 2026
    re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
               r"\s+\d{1,2},?\s+\d{4}\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
               r"[a-z]*\s+\d{4}\b", re.IGNORECASE),
]
_AMOUNT_RE = re.compile(
    r"(?:\$|€|£|USD|EUR|GBP)\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b"
    r"|\b\d{1,3}(?:,\d{3})+\.\d{2}\b"
)
# Capitalized sequences of 2+ words, excluding sentence starts is hard
# without NLP — we accept some noise and dedupe.
_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
_STOPWORDS = {"The", "This", "That", "These", "Those", "With", "From",
              "Page", "Table", "Figure", "Section"}


def extract_entities(text: str) -> dict:
    """Regex-based entity extraction: dates, amounts, candidate names."""
    dates: list[str] = []
    for rx in _DATE_RES:
        dates.extend(rx.findall(text))
    # findall with groups returns tuples; flatten defensively
    dates = [d if isinstance(d, str) else " ".join(d) for d in dates]

    amounts = _AMOUNT_RE.findall(text)

    names: list[str] = []
    for m in _NAME_RE.finditer(text):
        cand = m.group(1)
        if cand.split()[0] in _STOPWORDS:
            continue
        names.append(cand)

    def dedupe(seq: list[str]) -> list[str]:
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {
        "dates": dedupe(dates)[:20],
        "amounts": dedupe(amounts)[:20],
        "names": dedupe(names)[:20],
    }


# --- Document classification (heuristic keyword signals) ---

_CLASS_SIGNALS: dict[str, list[str]] = {
    "invoice": ["invoice", "bill to", "line item", "subtotal", "amount due",
                "payment terms", "invoice number", "purchase order"],
    "contract": ["agreement", "hereby", "party", "whereas", "indemnif",
                 "termination", "governing law", "exhibit a"],
    "report": ["executive summary", "findings", "methodology", "conclusion",
               "quarterly", "kpi", "year-over-year", "appendix"],
    "letter": ["dear ", "sincerely", "regards,", "to whom it may concern"],
}


def classify_document(text: str) -> str:
    """Heuristic document-type classification. Returns 'unknown' on ties."""
    lowered = text.lower()
    scores = {k: sum(1 for sig in sigs if sig in lowered)
              for k, sigs in _CLASS_SIGNALS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "unknown"
    # Tie → unknown (don't guess).
    if list(scores.values()).count(scores[best]) > 1:
        return "unknown"
    return best


# --- Ollama summarization ---

def summarize_ollama(text: str, model: str | None = None,
                     host: str | None = None,
                     timeout: int = 180) -> str:
    """Ask a local Ollama model for a concise business summary.

    Raises RuntimeError with a helpful message when Ollama is unreachable
    or returns nothing. Callers that must not crash should catch it.
    """
    host = (host or os.environ.get("OLLAMA_HOST",
                                   "http://localhost:11434")).rstrip("/")
    model = model or os.environ.get("OLLAMA_MODEL", "qwen3:8b")

    prompt = (
        "You are a business analyst. Read the document text below and produce "
        "a concise summary for a business user. Reply with JSON only:\n"
        '{"summary": "<3-5 sentence overview>", '
        '"key_points": ["<bullet>", "...up to 6 bullets>"]}\n\n'
        "DOCUMENT:\n" + text[:12000]  # keep prompt bounded
    )
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(
            f"Ollama not reachable at {host} (model '{model}'): {e}. "
            "Start it with 'ollama serve' and pull a model, e.g. "
            "'ollama pull qwen3:8b'."
        )
    raw = (data.get("response") or "").strip()
    if not raw:
        raise RuntimeError(f"Ollama returned an empty response (model '{model}').")
    return raw


def parse_summary_json(raw: str) -> dict:
    """Parse the LLM's JSON; tolerant of markdown fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"summary": cleaned[:2000], "key_points": []}
    return {
        "summary": str(parsed.get("summary", ""))[:2000],
        "key_points": [str(k) for k in parsed.get("key_points", [])][:8],
    }
