"""LLM summarization (Ollama-first) plus deterministic helpers.

- summarize_ollama: ask a local Ollama model for a business summary
- summarize_ollama_with_retry: same, with backoff on transient failures
- summarize_map_reduce: map-reduce summarization for long documents
- summarize_chunk: map step (summarize one chunk)
- extract_entities: regex dates, currency amounts, proper names
- classify_document: heuristic invoice | contract | report | letter | unknown
"""
from __future__ import annotations

import json
import os
import re
import time
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

# Bounds a single (non-streaming) /api/generate prompt. summarize_ollama
# truncates the document to this many chars; longer documents are handled
# by summarize_map_reduce instead (pipeline.py gates on this constant, so
# everything at or under the limit keeps the exact old code path).
SINGLE_CALL_CHAR_LIMIT = 12000

# Retry policy for transient Ollama failures (timeouts, connection resets).
# summarize_ollama keeps single-attempt legacy behavior; the pipeline uses
# the retrying variants below.
LLM_MAX_ATTEMPTS = 3
LLM_BACKOFF_SECONDS = (2, 5)


def _resolve_model(model: str | None) -> str:
    return model or os.environ.get("OLLAMA_MODEL", "qwen3:8b")


def _resolve_host(host: str | None) -> str:
    return (host or os.environ.get("OLLAMA_HOST",
                                  "http://localhost:11434")).rstrip("/")


def _summary_prompt(text: str) -> str:
    return (
        "You are a business analyst. Read the document text below and produce "
        "a concise summary for a business user. Reply with JSON only:\n"
        '{"summary": "<3-5 sentence overview>", '
        '"key_points": ["<bullet>", "...up to 6 bullets>"]}\n\n'
        "DOCUMENT:\n" + text[:SINGLE_CALL_CHAR_LIMIT]  # keep prompt bounded
    )


def _chunk_prompt(chunk: str) -> str:
    return (
        "You are a business analyst. Summarize the document excerpt below "
        "in 2-3 sentences and list up to 4 key points. Reply with JSON only:\n"
        '{"summary": "<2-3 sentence overview>", '
        '"key_points": ["<bullet>", "...up to 4 bullets>"]}\n\n'
        "EXCERPT:\n" + chunk
    )


def _reduce_prompt(combined: str) -> str:
    return (
        "You are a business analyst. Below are summaries of individual "
        "sections of a single document. Synthesize them into one coherent "
        "business summary for a business user. Reply with JSON only:\n"
        '{"summary": "<3-5 sentence overview>", '
        '"key_points": ["<bullet>", "...up to 6 bullets>"]}\n\n'
        "SECTION SUMMARIES:\n" + combined
    )


def _ollama_generate(prompt: str, model: str | None = None,
                     host: str | None = None, timeout: int = 180) -> str:
    """Single blocking /api/generate call. Raises RuntimeError on failure."""
    host = _resolve_host(host)
    model = _resolve_model(model)
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


def _generate_with_retry(prompt: str, model: str | None = None,
                         host: str | None = None, timeout: int = 180,
                         attempts: int = LLM_MAX_ATTEMPTS,
                         backoff: tuple = LLM_BACKOFF_SECONDS) -> str:
    """_ollama_generate with backoff on transient RuntimeErrors."""
    last: RuntimeError | None = None
    for i in range(max(1, attempts)):
        try:
            return _ollama_generate(prompt, model=model, host=host,
                                    timeout=timeout)
        except RuntimeError as e:
            last = e
            if i < attempts - 1:
                time.sleep(backoff[min(i, len(backoff) - 1)])
    assert last is not None
    raise last


def summarize_ollama(text: str, model: str | None = None,
                     host: str | None = None,
                     timeout: int = 180) -> str:
    """Ask a local Ollama model for a concise business summary.

    Single attempt (legacy behavior — unchanged). Callers needing
    resilience should use summarize_ollama_with_retry. Raises RuntimeError
    with a helpful message when Ollama is unreachable or returns nothing.
    Callers that must not crash should catch it.
    """
    return _ollama_generate(_summary_prompt(text), model=model,
                            host=host, timeout=timeout)


def summarize_ollama_with_retry(text: str, model: str | None = None,
                                host: str | None = None,
                                timeout: int = 180) -> str:
    """summarize_ollama with backoff on transient failures."""
    return _generate_with_retry(_summary_prompt(text), model=model,
                                host=host, timeout=timeout)


def summarize_chunk(chunk: str, model: str | None = None,
                    host: str | None = None, timeout: int = 180) -> dict:
    """Map step: summarize one chunk. Returns parsed {summary, key_points}."""
    raw = _generate_with_retry(_chunk_prompt(chunk), model=model,
                               host=host, timeout=timeout)
    return parse_summary_json(raw)


def _reduce_summaries(summaries: list[str], model: str | None,
                      host: str | None, timeout: int) -> dict:
    """Reduce step, applied hierarchically until the prompt fits."""
    combined = "\n\n".join(summaries)
    if len(summaries) <= 1:
        # Degenerate case: a single section summary always fits; truncate
        # defensively rather than recursing forever.
        combined = combined[:SINGLE_CALL_CHAR_LIMIT]
    if len(combined) <= SINGLE_CALL_CHAR_LIMIT:
        raw = _generate_with_retry(_reduce_prompt(combined), model=model,
                                   host=host, timeout=timeout)
        return parse_summary_json(raw)
    mid = max(1, len(summaries) // 2)
    left = _reduce_summaries(summaries[:mid], model, host, timeout)
    right = _reduce_summaries(summaries[mid:], model, host, timeout)
    return _reduce_summaries([left["summary"], right["summary"]],
                             model, host, timeout)


def summarize_map_reduce(text: str, chunks: list[str],
                         model: str | None = None,
                         host: str | None = None,
                         timeout: int = 180) -> dict:
    """Summarize a long document via map-reduce.

    Map: each chunk is summarized independently (small prompts, each well
    under the timeout); failed chunks are skipped and counted. Reduce:
    chunk summaries are synthesized with the standard business prompt,
    hierarchically if they don't fit one prompt. Raises RuntimeError if
    every chunk fails. Returns the parsed summary dict plus
    chunks_summarized / chunks_failed counts.
    """
    summaries: list[str] = []
    failed = 0
    for i, ch in enumerate(chunks):
        if not ch.strip():
            continue
        try:
            part = summarize_chunk(ch, model=model, host=host, timeout=timeout)
            key_pts = ("; ".join(part["key_points"])) if part["key_points"] else ""
            summaries.append(
                f"[Section {i + 1}] {part['summary']}"
                + (f" Key points: {key_pts}" if key_pts else "")
            )
        except RuntimeError:
            failed += 1
    if not summaries:
        raise RuntimeError(
            "map-reduce summarization failed: all chunks failed")
    result = _reduce_summaries(summaries, model, host, timeout)
    result["chunks_summarized"] = len(summaries)
    result["chunks_failed"] = failed
    return result


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
