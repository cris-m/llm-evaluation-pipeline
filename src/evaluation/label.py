import argparse
import json
from pathlib import Path


def load_results(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["results"]


def load_labels(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}

    labels = {}

    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            labels[row["id"]] = row["human_score"]

    return labels


def append_label(path: Path, result: dict, score: int) -> None:
    row = {
        "id": result["id"],
        "question": result["question"],
        "expected": result.get("expected"),
        "answer": result.get("answer", ""),
        "human_score": score,
    }

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_case(result: dict, number: int, total: int) -> None:
    print()
    print("=" * 70)
    print(f"Case {number}/{total}   {result['id']}")
    print("=" * 70)

    print("\nQuestion:")
    print(result["question"])

    print("\nReference:")
    print(result.get("expected") or "(no reference answer)")

    print("\nCandidate answer:")
    print(result.get("answer") or "(no answer)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human-label LLM judge evaluation results."
    )
    parser.add_argument(
        "results",
        type=Path,
        help="evaluation results JSON, e.g. runs/v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/calibration.jsonl"),
        help="where to save human labels",
    )
    args = parser.parse_args()

    results = load_results(args.results)
    labels = load_labels(args.output)

    cases = [
        result
        for result in results
        if result.get("expected") is not None
        and result.get("answer")
        and result["id"] not in labels
    ]

    if not cases:
        print("No unlabeled correctness cases found.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Results: {args.results}")
    print(f"Output:  {args.output}")
    print(f"Unlabeled cases: {len(cases)}")

    for number, result in enumerate(cases, start=1):
        print_case(result, number, len(cases))

        while True:
            value = input("\nHuman score [1-5, s=skip, q=quit]: ").strip().lower()

            if value == "q":
                print(f"\nStopped. Labels saved to {args.output}")
                return

            if value == "s":
                break

            if value in {"1", "2", "3", "4", "5"}:
                score = int(value)
                append_label(args.output, result, score)
                labels[result["id"]] = score
                print(f"Saved: {result['id']} -> {score}")
                break

            print("Enter 1, 2, 3, 4, 5, s, or q.")

    print(f"\nFinished. Labels saved to {args.output}")


if __name__ == "__main__":
    main()
