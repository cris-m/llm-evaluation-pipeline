import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from evaluation import llm_judge_scorers as judge
from evaluation.app import CONFIG_PATH, RagApp, load_config
from evaluation.code_evaluation import (
    DATASET_PATH,
    RESULTS_DIR,
    load_dataset,
    summarize,
)

JUDGES = {
    "correctness": judge.evaluate_correctness,
    "faithfulness": judge.evaluate_faithfulness,
}

FLAG_BELOW = {"correctness": 0.75, "faithfulness": 1.0}


def run_case(app: RagApp, example: dict) -> dict:
    record = {
        "id": example["id"],
        "question": example["question"],
        "tags": example.get("tags", []),
        "expected": example.get("expected"),
    }

    try:
        output = app.answer(example["question"])
    except Exception as error:
        return {**record, "error": f"{type(error).__name__}: {error}", "scores": {}}

    scores = {}
    for name, scorer in JUDGES.items():
        try:
            scores[name] = scorer(example, output)
        except Exception as error:
            scores[name] = {"score": None, "error": f"{type(error).__name__}: {error}"}

    return {
        **record,
        "answer": output["answer"],
        "sources": output.get("sources", []),
        "contexts": output["contexts"],
        "scores": scores,
    }


def print_summary(results: list[dict]) -> None:
    ok = [r for r in results if "error" not in r]
    summary = summarize(ok)

    print("\n" + "=" * 70)
    print(
        f"LLM-judge evaluation: {len(results)} cases, {len(results) - len(ok)} errors"
    )
    print("=" * 70)
    for name in JUDGES:
        unscored = sum(
            1
            for r in ok
            if r["scores"][name]["score"] is None
            and r["scores"][name].get("reason") != "no reference answer"
        )
        if name in summary:
            s = summary[name]
            print(
                f"{name:<16}{s['score']:>6.2f}  (n={s['count']}, unscored={unscored})"
            )

    print("\nBy tag, as mean (n):")
    print(f"{'tag':<14}" + "".join(f"{name:>20}" for name in JUDGES))
    for tag in sorted({t for r in ok for t in r["tags"]}):
        per_tag = summarize([r for r in ok if tag in r["tags"]])
        cells = [
            (
                f"{per_tag[n]['score']:.2f} ({per_tag[n]['count']})"
                if n in per_tag
                else "-"
            )
            for n in JUDGES
        ]
        print(f"{tag:<14}" + "".join(f"{cell:>20}" for cell in cells))

    print("\nCases to look at:")
    for r in results:
        if "error" in r:
            print(f"  {r['id']}: ERROR {r['error']}")
            continue
        for name, s in r["scores"].items():
            if s["score"] is None:
                if r["expected"] is not None or name != "correctness":
                    print(
                        f"  {r['id']}: {name} failed ({s.get('error') or s.get('reason')})"
                    )
            elif s["score"] < FLAG_BELOW[name]:
                print(f"  {r['id']}: {name}={s['score']:.2f}  {s.get('reason', '')}")
                for claim in s.get("unsupported", []):
                    print(f"      unsupported: {claim}")


def save_results(
    results: list[dict], run_name: str, output: Path | None, as_csv: bool
) -> list[Path]:
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = RESULTS_DIR / f"llm_judge_eval_{run_name}_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    meta = {"run": run_name, "judge_model": judge.MODEL, "judge_votes": judge.VOTES}
    payload = {"meta": meta, "results": results}
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    saved = [output]

    if as_csv:
        csv_path = output.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["id", "tags", *JUDGES, "error", "question", "expected", "answer"]
            )
            for r in results:
                scores = [(r["scores"].get(n) or {}).get("score") for n in JUDGES]
                writer.writerow(
                    [
                        r["id"],
                        ";".join(r["tags"]),
                        *scores,
                        r.get("error", ""),
                        r["question"],
                        r["expected"] or "",
                        r.get("answer", ""),
                    ]
                )
        saved.append(csv_path)
    return saved


def calibrate(path: Path) -> None:
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score

    ids, human, judged = [], [], []
    rows = load_dataset(path)
    missing = [r["id"] for r in rows if not isinstance(r.get("human_score"), int)]
    if missing:
        raise SystemExit(f"human_score must be an integer 1-5. Bad rows: {missing[:5]}")
    for row in rows:
        result = judge.evaluate_correctness(row, row)
        if result["score"] is None:
            print(f"{row['id']}: no score ({result['reason']})")
            continue
        ids.append(row["id"])
        human.append(row["human_score"])
        judged.append(round(result["score"] * 4 + 1))

    if not ids:
        raise SystemExit("The judge returned no scores.")

    n = len(ids)
    print(f"Judge calibration: {n} cases, judge {judge.MODEL}")
    print(f"exact agreement   {sum(h == j for h, j in zip(human, judged)) / n:.2f}")
    print(
        f"within 1 point    {sum(abs(h - j) <= 1 for h, j in zip(human, judged)) / n:.2f}"
    )
    print(f"spearman          {spearmanr(human, judged).statistic:.2f}")
    print(
        f"weighted kappa    {cohen_kappa_score(human, judged, weights='quadratic'):.2f}"
    )
    print(f"mean bias         {sum(j - h for h, j in zip(human, judged)) / n:+.2f}")

    print("\nDiffer by 2+ points:")
    for case_id, h, j in zip(ids, human, judged):
        if abs(h - j) >= 2:
            print(f"  {case_id}: human={h} judge={j}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the RAG app over the dataset and grade the answers with an LLM judge."
    )
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--tag", help="only run cases with this tag")
    parser.add_argument("--limit", type=int, help="only run the first N cases")
    parser.add_argument("--judge-model", help="default: judge_model from the config")
    parser.add_argument("--votes", type=int, default=judge.VOTES)
    parser.add_argument(
        "--calibrate", type=Path, help="check the judge against hand-graded answers"
    )
    parser.add_argument("--output", type=Path, help="save the JSON results here")
    parser.add_argument("--csv", action="store_true", help="also save a CSV")
    parser.add_argument(
        "--no-save", action="store_true", help="don't write any results file"
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than 0.")
    if args.votes <= 0:
        raise SystemExit("--votes must be greater than 0.")

    if args.judge_model:
        judge.MODEL = args.judge_model
    judge.VOTES = args.votes

    if args.calibrate:
        calibrate(args.calibrate)
        return

    config = load_config(args.config)
    if not args.judge_model and config.get("judge_model"):
        judge.MODEL = config["judge_model"]
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
    if config.get("model") == judge.MODEL:
        print(f"Warning: the app and the judge both use {judge.MODEL}.")

    results = []
    for index, example in enumerate(dataset, start=1):
        result = run_case(app, example)
        results.append(result)
        scored = [
            f"{n}={s['score']:.2f}"
            for n, s in result["scores"].items()
            if s["score"] is not None
        ]
        status = "ERROR" if "error" in result else " ".join(scored) or "no scores"
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
