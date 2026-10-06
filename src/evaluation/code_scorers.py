import re
from typing import Any

Example = dict[str, Any]
Output = dict[str, Any]

LATENCY_LIMIT_SECONDS = 15.0 


def _normalize(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text.strip().lower())


def _has_token(needle: str, text: str) -> bool:
    pattern = r"(?<![\w.,])" + re.escape(_normalize(needle)) + r"(?!\w|[.,]\d)"
    return re.search(pattern, text) is not None


def _answer(output: Output) -> str:
    return _normalize(output.get("answer", ""))


def _contexts(output: Output) -> list[str]:
    return [_normalize(context) for context in output.get("contexts", [])]


def must_contain(example: Example, output: Output) -> dict[str, Any]:
    required = example.get("must_contain") or []
    if not required:
        return {"score": None}

    answer = _answer(output)
    missing = [item for item in required if not _has_token(item, answer)]
    return {"score": 1 - len(missing) / len(required), "missing": missing}


REFUSAL_PATTERNS = [
    r"\bi don'?t know\b",
    r"\bi do not know\b",
    r"\b(doesn'?t|does not) (say|mention|specify|cover|contain)\b",
    r"\b(don'?t|do not) have (that |any |enough )?information\b",
    r"\bno (information|mention)\b",
    r"\bnot (provided|mentioned|specified|covered)\b",
    r"\b(cannot|can'?t|can not) (determine|find|answer)\b",
]

def is_refusal(answer: str) -> bool:
    text = _normalize(answer)
    return any(re.search(p, text) for p in REFUSAL_PATTERNS)


def abstains_correctly(example: Example, output: Output) -> dict[str, Any]:
    if "unanswerable" not in example.get("tags", []):
        return {"score": None}
    return {"score": float(is_refusal(output.get("answer", "")))}


def retrieval_hit(example: Example, output: Output) -> dict[str, Any]:
    evidence = example.get("evidence") or []
    if not evidence:
        return {"score": None}

    contexts = _contexts(output)
    hit = any(_normalize(item) in context for item in evidence for context in contexts)
    return {"score": float(hit)}


def evidence_recall(example: Example, output: Output) -> dict[str, Any]:
    evidence = example.get("evidence") or []
    if not evidence:
        return {"score": None}

    contexts = _contexts(output)
    missing = [
        item for item in evidence if not any(_normalize(item) in c for c in contexts)
    ]
    return {"score": 1 - len(missing) / len(evidence), "missing": missing}


def reciprocal_rank(example: Example, output: Output) -> dict[str, Any]:
    evidence = example.get("evidence") or []
    if not evidence:
        return {"score": None}

    for rank, context in enumerate(_contexts(output), start=1):
        if any(_normalize(item) in context for item in evidence):
            return {"score": 1.0 / rank, "rank": rank}
    return {"score": 0.0, "rank": None}


def latency_ok(
    example: Example, output: Output, limit: float = LATENCY_LIMIT_SECONDS
) -> dict[str, Any]:
    latency = output.get("latency")
    if latency is None:
        return {"score": None}
    return {"score": float(latency <= limit), "seconds": round(latency, 2)}


SCORERS = {
    "must_contain": must_contain,
    "abstains_correctly": abstains_correctly,
    "retrieval_hit": retrieval_hit,
    "evidence_recall": evidence_recall,
    "reciprocal_rank": reciprocal_rank,
    "latency_ok": latency_ok,
}
