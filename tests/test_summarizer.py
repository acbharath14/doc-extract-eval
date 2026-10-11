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
