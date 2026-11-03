"""Tests for summarizer: entities, classification, Ollama (mocked)."""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from doc_extract.summarizer import (
    extract_entities,
    classify_document,
    summarize_ollama,
    parse_summary_json,
)


def test_extract_dates():
    ents = extract_entities("Due on October 31, 2026. Signed 2026-10-01. "
                            "Also 10/11/2026 works.")
    assert "October 31, 2026" in ents["dates"]
    assert "2026-10-01" in ents["dates"]
    assert "10/11/2026" in ents["dates"]


def test_extract_amounts():
    ents = extract_entities("Total: $12,400.00. Tax €1,100.00. "
                            "Balance USD 500.00.")
    assert "$12,400.00" in ents["amounts"]
    assert any("1,100" in a for a in ents["amounts"])


def test_extract_names():
    ents = extract_entities("Contact Jane Smith at Acme Corporation "
                            "for details.")
    # Heuristic: may include a leading capitalized word ("Contact Jane Smith").
    assert any("Jane Smith" in n for n in ents["names"])
    assert "Acme Corporation" in ents["names"]


def test_extract_entities_empty():
    ents = extract_entities("")
    assert ents == {"dates": [], "amounts": [], "names": []}


def test_classify_invoice():
    text = ("Invoice INV-001. Bill to customer. Line items below. "
            "Subtotal $100. Amount due $108. Payment terms Net 30.")
    assert classify_document(text) == "invoice"


def test_classify_contract():
    text = ("This agreement is entered into by the parties. "
            "Whereas the parties agree, indemnification and termination "
            "clauses follow. Governing law applies.")
    assert classify_document(text) == "contract"


def test_classify_report():
    text = "Executive summary. Findings and methodology. Conclusion. Appendix."
    assert classify_document(text) == "report"


def test_classify_unknown_on_tie_or_empty():
    assert classify_document("hello world") == "unknown"
    assert classify_document("") == "unknown"


def _mock_urlopen(response_text: str):
    m = MagicMock()
    m.read.return_value = json.dumps({"response": response_text}).encode()
    m.__enter__.return_value = m
    return m


def test_summarize_ollama_success(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    body = json.dumps({"summary": "Q3 was strong.", "key_points": ["Revenue up"]})
    with patch("urllib.request.urlopen",
               return_value=_mock_urlopen(body)):
        out = summarize_ollama("some document text", model="qwen3:8b")
    parsed = json.loads(out)
    assert parsed["summary"] == "Q3 was strong."


def test_summarize_ollama_unreachable(monkeypatch):
    import urllib.error
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError("refused")):
        with pytest.raises(RuntimeError, match="not reachable"):
            summarize_ollama("text")


def test_parse_summary_json_plain():
    raw = '{"summary": "Hi.", "key_points": ["a", "b"]}'
    p = parse_summary_json(raw)
    assert p["summary"] == "Hi."
    assert p["key_points"] == ["a", "b"]


def test_parse_summary_json_markdown_fence():
    raw = '```json\n{"summary": "Hi.", "key_points": []}\n```'
    p = parse_summary_json(raw)
    assert p["summary"] == "Hi."


def test_parse_summary_json_garbage():
    p = parse_summary_json("not json at all {{{")
    assert p["key_points"] == []
    assert "not json" in p["summary"]


# --- Retry + map-reduce (new) ---

from doc_extract.summarizer import (
    summarize_chunk,
    summarize_map_reduce,
    summarize_ollama_with_retry,
    _generate_with_retry,
    _reduce_summaries,
    SINGLE_CALL_CHAR_LIMIT,
)


def _chunk_body(i: int) -> str:
    return json.dumps({"summary": f"Chunk {i} summary.",
                       "key_points": [f"point {i}"]})


def test_generate_with_retry_succeeds_after_transient_failure():
    import urllib.error
    ok = _mock_urlopen(json.dumps({"summary": "S.", "key_points": []}))
    with patch("urllib.request.urlopen",
               side_effect=[urllib.error.URLError("timeout"), ok]) as m:
        out = _generate_with_retry("prompt", model="m", backoff=(0, 0))
    assert json.loads(out)["summary"] == "S."
    assert m.call_count == 2


def test_generate_with_retry_exhausts_attempts():
    import urllib.error
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError("down")):
        with pytest.raises(RuntimeError, match="not reachable"):
            _generate_with_retry("prompt", model="m", attempts=2,
                                 backoff=(0, 0))


def test_summarize_ollama_with_retry_uses_single_bounded_prompt():
    seen = {}

    def fake(prompt, model=None, host=None, timeout=180, **kw):
        seen["prompt"] = prompt
        return '{"summary": "S.", "key_points": []}'

    with patch("doc_extract.summarizer._generate_with_retry",
               side_effect=fake):
        out = summarize_ollama_with_retry("x" * (SINGLE_CALL_CHAR_LIMIT + 500),
                                          model="m")
    assert out == '{"summary": "S.", "key_points": []}'
    # Truncation bound still respected on the single-call path.
    assert len(seen["prompt"]) < len("x" * (SINGLE_CALL_CHAR_LIMIT + 500)) + 2000


def test_summarize_chunk_success():
    body = json.dumps({"summary": "Excerpt about invoices.",
                       "key_points": ["Total $10"]})
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(body)):
        out = summarize_chunk("some excerpt", model="m")
    assert out["summary"] == "Excerpt about invoices."
    assert out["key_points"] == ["Total $10"]


def _fake_map_reduce_generate(fail_on=()):
    """Fake _generate_with_retry distinguishing map vs reduce prompts."""
    calls = {"n": 0}

    def fake(prompt, model=None, host=None, timeout=180, **kw):
        calls["n"] += 1
        if "EXCERPT:" in prompt:
            idx = calls["n"]
            if idx in fail_on:
                raise RuntimeError("chunk boom")
            return _chunk_body(idx)
        assert "SECTION SUMMARIES:" in prompt
        return json.dumps({"summary": "Final summary.",
                           "key_points": ["final point"]})

    return fake, calls


def test_map_reduce_happy_path():
    fake, calls = _fake_map_reduce_generate()
    chunks = ["chunk one text", "chunk two text", "chunk three text"]
    with patch("doc_extract.summarizer._generate_with_retry",
               side_effect=fake):
        out = summarize_map_reduce("x" * (SINGLE_CALL_CHAR_LIMIT + 1),
                                   chunks, model="m")
    assert out["summary"] == "Final summary."
    assert out["key_points"] == ["final point"]
    assert out["chunks_failed"] == 0
    assert out["chunks_summarized"] == 3
    assert calls["n"] == 4  # 3 map + 1 reduce


def test_map_reduce_skips_failed_chunks():
    fake, calls = _fake_map_reduce_generate(fail_on={2})
    chunks = ["c1", "c2", "c3"]
    with patch("doc_extract.summarizer._generate_with_retry",
               side_effect=fake):
        out = summarize_map_reduce("x" * (SINGLE_CALL_CHAR_LIMIT + 1),
                                   chunks, model="m")
    assert out["summary"] == "Final summary."
    assert out["chunks_failed"] == 1
    assert out["chunks_summarized"] == 2


def test_map_reduce_all_chunks_fail():
    def fake(prompt, model=None, host=None, timeout=180, **kw):
        raise RuntimeError("always fails")

    with patch("doc_extract.summarizer._generate_with_retry",
               side_effect=fake):
        with pytest.raises(RuntimeError, match="all chunks failed"):
            summarize_map_reduce("x" * (SINGLE_CALL_CHAR_LIMIT + 1),
                                 ["c1", "c2"], model="m")


def test_reduce_summaries_hierarchical():
    calls = []

    def fake(prompt, model=None, host=None, timeout=180, **kw):
        calls.append(len(prompt))
        return json.dumps({"summary": "R.", "key_points": []})

    with patch("doc_extract.summarizer._generate_with_retry",
               side_effect=fake):
        out = _reduce_summaries(["y" * 9000] * 4, "m", None, 180)
    assert out["summary"] == "R."
    # No prompt sent over the limit; recursion absorbed the overflow.
    assert all(c <= SINGLE_CALL_CHAR_LIMIT + 2000 for c in calls)
    assert len(calls) == 7  # 4 leaves + 2 halves + 1 final
