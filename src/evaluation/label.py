import argparse
import json
from pathlib import Path


def load_results(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "results" not in data:
        raise SystemExit("Expected a judge run ({'meta', 'results'}).")
    return data["results"]


def load_labels(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    labels = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                labels[row["id"]] = row["human_score"]
    return labels


def append_label(path: Path, result: dict, score: int, answer: str | None = None, row_id: str | None = None) -> None:
    row = {
        "id": row_id or result["id"],
        "question": result["question"],
        "expected": result.get("expected"),
        "answer": answer if answer is not None else result.get("answer", ""),
        "human_score": score,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_case(result: dict, number: int, total: int) -> None:
    print("\n" + "=" * 70)
    print(f"Case {number}/{total}   {result['id']}")
    print("=" * 70)
    print("\nQuestion:\n" + result["question"])
    print("\nReference:\n" + (result.get("expected") or "(no reference answer)"))
    print("\nCandidate answer:\n" + (result.get("answer") or "(no answer)"))


def grade(prompt: str) -> int | None:
    while True:
        value = input(prompt).strip().lower()
        if value in {"1", "2", "3", "4", "5"}:
            return int(value)
        if value == "":
            return None
        print("Enter 1-5 (or press Enter to cancel).")


def spread(labels: dict[str, int]) -> dict[int, int]:
    values = list(labels.values())
    return {s: values.count(s) for s in range(1, 6)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-label LLM judge evaluation results.")
    parser.add_argument("results", type=Path, help="judge results JSON, e.g. runs/v2.json")
    parser.add_argument("--output", type=Path, default=Path("datasets/calibration.jsonl"))
    args = parser.parse_args()

    results = load_results(args.results)
    labels = load_labels(args.output)

    cases = [
        r for r in results
        if r.get("expected") is not None and r.get("answer") and r["id"] not in labels
    ]
    if not cases:
        print("No unlabeled correctness cases found.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Results: {args.results}\nOutput:  {args.output}\nUnlabeled cases: {len(cases)}")
    print(
        "\nKeys: 1-5 = your grade | w = add a wrong or partial answer of your own to grade"
        " | s = skip | q = quit"
    )

    for number, result in enumerate(cases, start=1):
        print_case(result, number, len(cases))

        while True:
            value = input("\nHuman score [1-5, w, s, q]: ").strip().lower()

            if value == "q":
                print(f"\nStopped. Labels saved to {args.output}")
                print(f"Scores so far: {spread(labels)}")
                return

            if value == "s":
                break

            if value == "w":
                wrong = input("Write a wrong or partial answer: ").strip()
                if not wrong:
                    continue
                score = grade("Grade for that answer [1-5, Enter to cancel]: ")
                if score is None:
                    continue
                n = sum(1 for k in labels if k.startswith(f"{result['id']}-w")) + 1
                row_id = f"{result['id']}-w{n}"
                append_label(args.output, result, score, answer=wrong, row_id=row_id)
                labels[row_id] = score
                print(f"Saved: {row_id} -> {score}")
                continue  # same case again: grade the real answer next

            if value in {"1", "2", "3", "4", "5"}:
                score = int(value)
                append_label(args.output, result, score)
                labels[result["id"]] = score
                print(f"Saved: {result['id']} -> {score}")
                break

            print("Enter 1, 2, 3, 4, 5, w, s, or q.")

    print(f"\nFinished. Labels saved to {args.output}")
    counts = spread(labels)
    print(f"Scores so far: {counts}")
    if len(labels) < 30:
        print("Aim for at least 30 labels.")
    if labels and max(counts.values()) > 0.5 * len(labels):
        print("Most labels share one score. Add wrong/partial answers with 'w', or kappa will be meaningless.")


if __name__ == "__main__":
    main()