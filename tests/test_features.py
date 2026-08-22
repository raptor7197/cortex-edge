"""Tests for feature extraction and task classification."""

from app.router.features import classify_task, extract_features


def test_greeting_is_simple():
    f = extract_features("Hello!")
    assert f.token_count == 1
    assert f.complexity_score < 0.35


def test_code_query_detected():
    f = extract_features("Write a Python function to sort a list")
    assert f.has_code == 1
    assert f.complexity_score >= 0.15  # code weight enters the score


def test_reasoning_query_high_complexity():
    f = extract_features(
        "Derive the formula A = P(1 + r)^n and write Python code to compute it."
    )
    assert f.has_code == 1
    assert f.has_math == 1
    assert f.reasoning_score > 0.0
    assert f.complexity_score >= 0.45


def test_documents_raise_complexity():
    plain = extract_features("Summarize the paper")
    with_docs = extract_features("Summarize the paper", requires_documents=True)
    assert with_docs.requires_documents == 1
    assert with_docs.complexity_score > plain.complexity_score


def test_math_detected():
    f = extract_features("Solve for x: 2x + 5 = 13")
    assert f.has_math == 1


def test_classify_task():
    assert classify_task("Hello there") == "greeting"
    assert classify_task("Write python code") == "code"
    assert classify_task("What is the capital?") == "simple"