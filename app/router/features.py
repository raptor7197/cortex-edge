"""Query feature & complexity extraction (PLAN 15.5).

Prompt features feed the rule router, the learned router, and the
dataset builder. They replace naive length-only complexity heuristics.
"""

import math
import re
from dataclasses import asdict, dataclass

REASONING_WORDS = {"derive", "prove", "compare", "analyse", "analyze", "why", "steps"}
CODE_MARKERS = {"python", "java", "c++", "code", "function", "algorithm", "debug"}
MATH_PATTERN = re.compile(r"[=+\-*/^]|\b(sin|cos|log|integral|matrix)\b", re.I)

# Lightweight task-class keywords used by the rule router / dataset labels.
TASK_KEYWORDS = {
    "greeting": {"hello", "hi", "hey", "thanks", "thank", "bye"},
    "code": CODE_MARKERS,
    "math": {"calculate", "compute", "solve", "equation", "sum", "math", "formula"},
    "document": {"document", "paper", "file", "upload", "pdf", "article", "manual"},
}


@dataclass
class QueryFeatures:
    token_count: int
    sentence_count: int
    has_code: int
    has_math: int
    reasoning_score: float
    multi_part_score: float
    requires_documents: int
    complexity_score: float

    def as_dict(self) -> dict:
        return asdict(self)


def extract_features(text: str, requires_documents: bool = False) -> QueryFeatures:
    words = re.findall(r"\b\w+\b", text.lower())
    token_count = len(words)
    sentence_count = max(1, len(re.findall(r"[.!?]", text)))
    has_code = int(any(w in CODE_MARKERS for w in words) or "```" in text)
    has_math = int(bool(MATH_PATTERN.search(text)))
    reasoning_hits = sum(w in REASONING_WORDS for w in words)
    reasoning_score = min(1.0, reasoning_hits / 3.0)
    multi_part_score = min(1.0, (text.count("?") + text.count(";") + text.count("\n")) / 4.0)
    length_score = min(1.0, math.log1p(token_count) / math.log(513))
    complexity = (
        0.20 * length_score + 0.20 * reasoning_score + 0.15 * has_code
        + 0.15 * has_math + 0.15 * multi_part_score + 0.15 * int(requires_documents)
    )
    return QueryFeatures(
        token_count, sentence_count, has_code, has_math,
        reasoning_score, multi_part_score,
        int(requires_documents), round(complexity, 4),
    )


def classify_task(text: str) -> str:
    """Coarse task class (greeting/simple/code/math/document/complex)."""
    words = set(re.findall(r"\b\w+\b", text.lower()))
    for cls, keywords in TASK_KEYWORDS.items():
        if words & keywords:
            return cls
    feats = extract_features(text)
    if feats.complexity_score >= 0.70:
        return "complex"
    return "simple"
