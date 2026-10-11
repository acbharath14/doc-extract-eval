"""Tests for ground-truth scoring."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from doc_extract.scoring import (
    normalize_text,
    score_against_ground_truth,
    load_ground_truth,
)
from doc_extract.metrics import pick_winner, compare_extractors


def test_normalize_text():
    assert normalize_text("Hello, WORLD!") == ["hello", "world"]
    assert normalize_text("don't stop") == ["don't", "stop"]
    assert normalize_text("  spaces\tand\nnewlines  ") == ["spaces", "and", "newlines"]
    assert normalize_text("$1,234.56") == ["1", "234", "56"]


def test_perfect_score():
    s = score_against_ground_truth("The quick brown fox", "the QUICK brown fox!")
    assert s["precision"] == 1.0
    assert s["recall"] == 1.0
    assert s["f1"] == 1.0
    assert s["n_correct"] == 4


def test_partial_recall():
    # Extractor found half the words, all correct.
    s = score_against_ground_truth("quick brown", "the quick brown fox")
    assert s["precision"] == 1.0
    assert s["recall"] == 0.5
    assert s["f1"] == round(2 * 1.0 * 0.5 / 1.5, 4)


def test_low_precision():
    # Extractor hallucinated extra words.
    s = score_against_ground_truth(
        "the quick brown fox jumps over", "the quick brown fox")
    assert s["recall"] == 1.0
    assert s["precision"] == round(4 / 6, 4)
    assert s["f1"] < 1.0


def test_empty_extraction():
    s = score_against_ground_truth("", "some expected text")
    assert s["precision"] == 0.0
    assert s["recall"] == 0.0
    assert s["f1"] == 0.0


def test_empty_ground_truth():
    s = score_against_ground_truth("some text", "")
    assert s["precision"] == 0.0
    assert s["recall"] == 0.0


def test_repeated_words_count():
    # Multiset: "fox fox" vs "fox" → recall 0.5, not 1.0.
    s = score_against_ground_truth("fox", "fox fox")
    assert s["recall"] == 0.5
    assert s["n_correct"] == 1


def test_load_ground_truth_explicit(tmp_path):
    gt = tmp_path / "expected.txt"
    gt.write_text("hello world")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    assert load_ground_truth(str(pdf), str(gt)) == "hello world"


def test_load_ground_truth_sibling(tmp_path):
    (tmp_path / "doc.txt").write_text("sibling truth")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    assert load_ground_truth(str(pdf)) == "sibling truth"


def test_load_ground_truth_missing(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    assert load_ground_truth(str(pdf)) is None


def test_pick_winner_uses_f1_with_ground_truth():
    scored = [
        {"extractor": "verbose", "char_count": 1000, "n_tables": 0,
         "elapsed_s": 0.1, "n_warnings": 0,
         "gt": {"f1": 0.5, "precision": 0.4, "recall": 0.7}},
        {"extractor": "precise", "char_count": 500, "n_tables": 0,
         "elapsed_s": 0.1, "n_warnings": 0,
         "gt": {"f1": 0.9, "precision": 0.95, "recall": 0.85}},
    ]
    w = pick_winner(scored)
    assert w["winner"] == "precise"
    assert w["scored_by"] == "ground_truth_f1"
    assert "F1" in w["reason"]


def test_pick_winner_falls_back_without_ground_truth():
    scored = [
        {"extractor": "a", "char_count": 100, "n_tables": 0,
         "elapsed_s": 0.1, "n_warnings": 0},
        {"extractor": "b", "char_count": 200, "n_tables": 0,
         "elapsed_s": 0.1, "n_warnings": 0},
    ]
    w = pick_winner(scored)
    assert w["winner"] == "b"
    assert w["scored_by"] == "char_count"


def test_compare_with_ground_truth_flag(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent.parent / "samples"))
    from text_pdf import build
    pdf = tmp_path / "r.pdf"
    build(pdf)
    # Ground truth that matches the generated report's key phrases.
    gt = "quarterly business report revenue growth"
    out = compare_extractors(str(pdf), ground_truth=gt)
    assert out["ground_truth_used"] is True
    assert out["winner"]["scored_by"] == "ground_truth_f1"
    for m in out["results"].values():
        assert "gt" in m
        assert 0.0 <= m["gt"]["f1"] <= 1.0
