import argparse
import json
import random
import statistics
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from evaluation import llm_judge_scorers as judge
from evaluation.app import CONFIG_PATH, load_config
from evaluation.code_evaluation import summarize
from evaluation.code_scorers import is_refusal
from evaluation.compare import load_run
from evaluation.trace import find_run_file

BASE_DIR = Path(__file__).resolve().parent
LOGS_PATH = BASE_DIR / "logs" / "production.jsonl"


def shown(path: Path) -> str:
    return (
        str(path.relative_to(BASE_DIR)) if path.is_relative_to(BASE_DIR) else str(path)
    )


def load_logs(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"No log file at {shown(path)}.\n"
            "Log one JSON line per request ({question, answer, contexts, thumbs_down?}), "
            "or try the script on a saved run with --from-run runs/v4.json"
        )
    logs = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path.name}, line {number}: invalid JSON ({error})")
            if entry.get("question") and entry.get("answer"):
                logs.append({"log_id": f"log{number:04d}", **entry})
    return logs


def recent(logs: list[dict], days: float) -> list[dict]:
    cutoff = datetime.now() - timedelta(days=days)
    kept = []
    for entry in logs:
        try:
            when = datetime.fromisoformat(entry["ts"])
        except (KeyError, TypeError, ValueError):
            kept.append(entry)
            continue
        if when >= cutoff:
            kept.append(entry)
    return kept


def logs_from_run(value: str) -> list[dict]:
    run = load_run(find_run_file(value))
    return [
        {
            "log_id": f"run-{r['id']}",
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r.get("contexts", []),
            "latency": r.get("latency"),
        }
        for r in run["results"]
        if "error" not in r and r.get("answer")
    ]


def interval(values: list[float], n_boot: int = 2000, level: float = 95.0):
    rng = np.random.default_rng(0)
    data = np.array(values)
    means = data[rng.integers(0, len(data), size=(n_boot, len(data)))].mean(axis=1)
    tail = (100 - level) / 2
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def baseline_faithfulness(value: str) -> float:
    run = load_run(find_run_file(value))
    ok = [r for r in run["results"] if "error" not in r]
    summary = summarize(ok).get("faithfulness")
    if summary is None:
        raise SystemExit(
            f"{value}: this run has no faithfulness scores to use as a baseline."
        )
    return summary["score"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a sample of real traffic for faithfulness (no reference answer needed)."
    )
    parser.add_argument(
        "--logs",
        type=Path,
        default=LOGS_PATH,
        help=f"request log (default: {shown(LOGS_PATH)})",
    )
    parser.add_argument(
        "--from-run", help="use a saved run as the log (a smoke test of this script)"
    )
    parser.add_argument(
        "--days",
        type=float,
        help="only requests from the last N days (needs a ts on each line)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=30,
        help="how many requests to score (default: 30)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config", default=CONFIG_PATH, help="read judge_model from this config"
    )
    parser.add_argument("--judge-model", help="default: judge_model from the config")
    parser.add_argument(
        "--baseline", help="offline run (file, name or path) to compare against"
    )
    parser.add_argument(
        "--margin", type=float, default=0.10, help="alert when below baseline - margin"
    )
    parser.add_argument(
        "--output", type=Path, help="save the per-request scores as JSON"
    )
    parser.add_argument(
        "--history",
        type=Path,
        help="append one summary line per run to this file, to track it over time",
    )
    args = parser.parse_args()

    if args.sample <= 0:
        raise SystemExit("--sample must be greater than 0.")

    logs = logs_from_run(args.from_run) if args.from_run else load_logs(args.logs)
    if args.days:
        logs = recent(logs, args.days)
    if not logs:
        raise SystemExit(
            "No usable log entries (each needs a question and an answer, and fall in the --days window)."
        )

    config = load_config(args.config)
    judge.MODEL = args.judge_model or config.get("judge_model") or judge.MODEL

    sample = random.Random(args.seed).sample(logs, min(args.sample, len(logs)))
    print(f"{len(logs)} logged requests, scoring {len(sample)} with {judge.MODEL}\n")

    scored, failed = [], []
    for index, entry in enumerate(sample, start=1):
        try:
            result = judge.evaluate_faithfulness(
                {"question": entry["question"]},
                {"answer": entry["answer"], "contexts": entry.get("contexts", [])},
            )
        except Exception as exc:
            failed.append(entry["log_id"])
            print(
                f"[{index}/{len(sample)}] {entry['log_id']}  ERROR {type(exc).__name__}: {exc}"
            )
            continue
        if result["score"] is None:
            failed.append(entry["log_id"])
            print(
                f"[{index}/{len(sample)}] {entry['log_id']}  not scored ({result.get('reason')})"
            )
            continue
        scored.append(
            {"log_id": entry["log_id"], "question": entry["question"], **result}
        )
        print(
            f"[{index}/{len(sample)}] {entry['log_id']}  faithfulness={result['score']:.2f}"
        )

    print("\n" + "=" * 70)
    print(f"Online evaluation: {len(scored)} scored, {len(failed)} not scored")
    print("=" * 70)

    alert = False
    mean = low = high = base = None
    if scored:
        values = [s["score"] for s in scored]
        mean = statistics.mean(values)
        low, high = interval(values)
        print(
            f"faithfulness            {mean:.2f}   (95% interval {low:.2f} to {high:.2f}, n={len(values)})"
        )

        if args.baseline:
            base = baseline_faithfulness(args.baseline)
            floor = base - args.margin
            alert = mean < floor
            print(f"offline baseline        {base:.2f}   (alert below {floor:.2f})")
            print(
                "ALERT: production faithfulness dropped below the baseline."
                if alert
                else "Within the baseline range."
            )
    else:
        alert = True
        print("ALERT: no request could be scored. Is the judge model running?")

    refusals = sum(is_refusal(entry["answer"]) for entry in logs)
    rate_down = None
    latency_mean = None
    print(f"\nAll {len(logs)} logged requests:")
    print(
        f"  abstention rate       {refusals / len(logs):.2f}  (a sudden jump or drop is a warning sign)"
    )
    flagged = [entry for entry in logs if "thumbs_down" in entry]
    if flagged:
        down = sum(bool(entry["thumbs_down"]) for entry in flagged)
        rate_down = down / len(flagged)
        print(
            f"  thumbs-down rate      {down / len(flagged):.2f}  ({down} of {len(flagged)} rated)"
        )
    latencies = [
        e["latency"] for e in logs if isinstance(e.get("latency"), (int, float))
    ]
    if latencies:
        latency_mean = statistics.mean(latencies)
        print(
            f"  latency               mean {statistics.mean(latencies):.1f}s, max {max(latencies):.1f}s"
        )

    worst = sorted((s for s in scored if s["score"] < 1), key=lambda s: s["score"])[:5]
    if worst:
        print("\nLeast faithful answers in the sample:")
        for s in worst:
            print(f"  {s['log_id']}  {s['score']:.2f}  {s['question']!r}")
            for claim in s.get("unsupported", [])[:3]:
                print(f"      unsupported: {claim}")

    if args.history:
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "requests": len(logs),
            "scored": len(scored),
            "faithfulness": mean,
            "ci_low": low,
            "ci_high": high,
            "baseline": base,
            "alert": alert,
            "abstention_rate": refusals / len(logs),
            "thumbs_down_rate": rate_down,
            "latency_mean": latency_mean,
        }
        args.history.parent.mkdir(parents=True, exist_ok=True)
        with args.history.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        print(f"\nHistory: added a line to {shown(args.history)}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"judge": judge.MODEL, "scored": scored, "not_scored": failed},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved: {args.output}")

    if alert:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
