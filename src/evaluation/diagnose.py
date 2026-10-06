import argparse
import json
from collections import Counter
from pathlib import Path

from evaluation.code_evaluation import DATASET_PATH, load_dataset
from evaluation.code_scorers import evidence_recall, is_refusal

CORRECT_AT = 0.75  # a judge score of 4 out of 5 or better
FAITHFUL_AT = 0.5


def score(result: dict, name: str) -> float | None:
    return result["scores"].get(name, {}).get("score")


def diagnose(example: dict, result: dict) -> str:
    if "error" in result:
        return "app error"

    refused = is_refusal(result["answer"])

    if example["expected"] is None:
        return "ok" if refused else "answered unanswerable"

    correct = score(result, "correctness")
    if correct is None:
        return "judge failed"
    if correct >= CORRECT_AT:
        return "ok"

    recall = evidence_recall(example, result)["score"]
    if recall == 0:
        return "retrieval miss"
    if recall < 1:
        return "retrieval partial"

    if refused:
        return "over-abstained"
    faithful = score(result, "faithfulness")
    if faithful is not None and faithful < FAITHFUL_AT:
        return "unfaithful"
    return "incomplete or misread"


def show_failure(example: dict, result: dict, label: str, verbose: bool) -> None:
    print(
        f"  {result['id']}: {label}  "
        f"correctness={score(result, 'correctness')}  "
        f"faithfulness={score(result, 'faithfulness')}"
    )
    if label == "app error":
        print(f"      error: {result['error']}")
        return

    if label.startswith("retrieval"):
        for quote in evidence_recall(example, result).get("missing", []):
            print(f"      missing evidence: {quote}")

    for claim in result["scores"].get("faithfulness", {}).get("unsupported", []):
        print(f"      unsupported: {claim}")

    if example["expected"] is None:
        print(f"      answer: {result['answer'][:150]!r}")

    if verbose:
        print(f"      sources: {', '.join(result['sources']) or 'none'}")
        if example["expected"] is not None:
            print(f"      judge: {result['scores']['correctness'].get('reason')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label each case in a judge run with the stage most likely to have failed."
    )
    parser.add_argument(
        "results",
        type=Path,
        help="judge results JSON, e.g. runs/v2.json",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also show retrieved sources and the judge's reasoning",
    )
    args = parser.parse_args()

    run = json.loads(args.results.read_text(encoding="utf-8"))
    if not isinstance(run, dict) or "results" not in run:
        raise SystemExit(
            "Expected a judge run ({'meta', 'results'}). Code-only runs have no judge scores."
        )
    dataset = {row["id"]: row for row in load_dataset(args.dataset)}
    stale = [r["id"] for r in run["results"] if r["id"] not in dataset]
    if stale:
        print(
            f"Skipping {len(stale)} cases not in the dataset anymore: {', '.join(stale)}\n"
        )
    run["results"] = [r for r in run["results"] if r["id"] in dataset]
    labels = {r["id"]: diagnose(dataset[r["id"]], r) for r in run["results"]}

    for label, count in Counter(labels.values()).most_common():
        print(f"{label:<24}{count}")

    print("\nFailures:")
    for r in run["results"]:
        label = labels[r["id"]]
        if label != "ok":
            show_failure(dataset[r["id"]], r, label, args.verbose)

    shaky = []
    for r in run["results"]:
        faithful = score(r, "faithfulness")
        if labels[r["id"]] == "ok" and faithful is not None and faithful < 1:
            shaky.append(r["id"])
    print(f"\nCorrect, but faithfulness below 1.00: {', '.join(shaky) or 'none'}")


if __name__ == "__main__":
    main()
