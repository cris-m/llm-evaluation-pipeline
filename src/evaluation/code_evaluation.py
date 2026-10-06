import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from evaluation.app import CONFIG_PATH, RagApp, load_config
from evaluation.code_scorers import SCORERS

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datasets" / "dataset.jsonl"
RESULTS_DIR = BASE_DIR / "results"


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name}, line {number}: invalid JSON ({error})"
                ) from error
            for key in ("id", "question"):
                if key not in row:
                    raise ValueError(f"{path.name}, line {number}: missing '{key}'")
            rows.append(row)
    return rows


def apply_scorers(example: dict, output: dict) -> dict:
    scores = {}
    for name, scorer in SCORERS.items():
        try:
            scores[name] = scorer(example, output)
        except Exception as error:
            scores[name] = {"score": None, "error": f"{type(error).__name__}: {error}"}
    return scores


def run_case(app: RagApp, example: dict) -> dict:
    record = {
        "id": example["id"],
        "question": example["question"],
        "tags": example.get("tags", []),
    }

    start = time.perf_counter()
    try:
        output = app.answer(example["question"])
        output["latency"] = time.perf_counter() - start
        answer, contexts = output["answer"], output["contexts"]
    except Exception as error:
        return {**record, "error": f"{type(error).__name__}: {error}", "scores": {}}

    return {
        **record,
        "answer": answer,
        "sources": output.get("sources", []),
        "contexts": contexts,
        "latency": output["latency"],
        "scores": apply_scorers(example, output),
    }


def summarize(results: list[dict]) -> dict:
    values = defaultdict(list)
    for result in results:
        for name, scored in result["scores"].items():
            if scored["score"] is not None:
                values[name].append(scored["score"])
    return {
        name: {"score": statistics.mean(v), "count": len(v)}
        for name, v in values.items()
    }


def print_summary(results: list[dict]) -> None:
    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    summary = summarize(ok)

    print("\n" + "=" * 70)
    print(f"Code-based evaluation: {len(results)} cases, {len(errors)} errors")
    print("=" * 70)
    for name, result in summary.items():
        print(f"{name:<22}{result['score']:>6.2f}  (n={result['count']})")

    print("\nBy tag, as mean (n):")
    print(f"{'tag':<14}" + "".join(f"{name:>20}" for name in summary))
    for tag in sorted({t for r in ok for t in r["tags"]}):
        per_tag = summarize([r for r in ok if tag in r["tags"]])
        cells = [
            (
                f"{per_tag[n]['score']:.2f} ({per_tag[n]['count']})"
                if n in per_tag
                else "-"
            )
            for n in summary
        ]
        print(f"{tag:<14}" + "".join(f"{cell:>20}" for cell in cells))

    if ok:
        latencies = [r["latency"] for r in ok]
        print(
            f"\nLatency: mean {statistics.mean(latencies):.1f}s, max {max(latencies):.1f}s"
        )

    print("\nCases to look at (any score below 1, a broken scorer, or an error):")
    for r in results:
        if "error" in r:
            print(f"  {r['id']}: ERROR {r['error']}")
            continue
        problems = [
            f"{n}(ERROR)" if "error" in s else n
            for n, s in r["scores"].items()
            if "error" in s or (s["score"] is not None and s["score"] < 1)
        ]
        if problems:
            print(f"  {r['id']}: {', '.join(problems)} | {r['answer'][:60]!r}")


def save_results(
    results: list[dict], run_name: str, output: Path | None, as_csv: bool
) -> list[Path]:
    if output is None:
        output = (
            RESULTS_DIR / f"code_eval_{run_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    saved = [output]

    if as_csv:
        csv_path = output.with_suffix(".csv")
        names = list(SCORERS)
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["id", "tags", "latency", *names, "error", "question", "answer"]
            )
            for r in results:
                cells = [
                    (r["scores"].get(n) or {}).get("score") for n in names
                ]
                writer.writerow(
                    [
                        r["id"],
                        ";".join(r["tags"]),
                        round(r.get("latency", 0), 2),
                        *cells,
                        r.get("error", ""),
                        r["question"],
                        r.get("answer", ""),
                    ]
                )
        saved.append(csv_path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the RAG app over the dataset with the code scorers."
    )
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--tag", help="only run cases with this tag")
    parser.add_argument("--limit", type=int, help="only run the first N cases")
    parser.add_argument(
        "--output",
        type=Path,
        help="save the JSON results here (default: results/ folder)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="also save a one-row-per-case CSV next to the JSON",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="don't write any results file"
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than 0.")

    config = load_config(args.config)
    dataset = load_dataset(args.dataset)
    if args.tag:
        dataset = [e for e in dataset if args.tag in e.get("tags", [])]
    dataset = dataset[: args.limit]
    if not dataset:
        raise SystemExit("No cases to run (check --tag and --limit).")

    app = RagApp(config)
    print(
        f"{config.get('name', 'run')}: {len(app.chunks)} chunks, {len(dataset)} cases"
    )

    results = []
    for index, example in enumerate(dataset, start=1):
        result = run_case(app, example)
        results.append(result)
        scored = [
            f"{n}={s['score']:.2f}"
            for n, s in result["scores"].items()
            if s["score"] is not None
        ]
        status = "ERROR" if "error" in result else " ".join(scored)
        print(f"[{index}/{len(dataset)}] {example['id']}  {status}")

    print_summary(results)

    if not args.no_save:
        print()
        for path in save_results(
            results, config.get("name", "run"), args.output, args.csv
        ):
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
