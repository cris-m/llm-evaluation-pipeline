from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation import llm_judge_scorers as judge
from evaluation.app import CONFIG_PATH, RagApp, load_config
from evaluation.code_evaluation import (
    DATASET_PATH,
    RESULTS_DIR,
    load_dataset,
    summarize,
)
from evaluation.code_scorers import SCORERS as CODE_SCORERS

JUDGE_SCORERS = {
    "correctness": judge.evaluate_correctness,
    "faithfulness": judge.evaluate_faithfulness,
}

ALL_SCORERS = {
    **CODE_SCORERS,
    **JUDGE_SCORERS,
}


def parse_scorers(text: str) -> dict[str, Any]:
    if text.strip().lower() == "all":
        return dict(ALL_SCORERS)

    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in ALL_SCORERS]

    if unknown:
        available = ", ".join(ALL_SCORERS)
        raise SystemExit(
            f"Unknown scorer(s): {', '.join(unknown)}\n" f"Available: {available}"
        )

    return {name: ALL_SCORERS[name] for name in names}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def scrub_home_paths(text: str) -> str:
    home = str(Path.home())
    return text.replace(home, "~")


def run_case(
    app: RagApp,
    example: dict[str, Any],
    scorers: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "id": example["id"],
        "question": example["question"],
        "tags": example.get("tags", []),
        "expected": example.get("expected"),
    }

    started = time.perf_counter()

    try:
        output = app.answer(example["question"])
    except Exception as exc:
        return {
            **record,
            "error": f"{type(exc).__name__}: {exc}",
            "scores": {},
        }

    latency = time.perf_counter() - started

    scores: dict[str, Any] = {}

    for name, scorer in scorers.items():
        try:
            scores[name] = scorer(example, output)
        except Exception as exc:
            scores[name] = {
                "score": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {
        **record,
        "answer": output["answer"],
        "sources": output.get("sources", []),
        "contexts": output.get("contexts", []),
        "latency": round(latency, 3),
        "scores": scores,
    }


def scorer_failed(score: dict[str, Any]) -> bool:
    if "error" in score:
        return True

    return score.get("score") is None and score.get("reason") not in (
        None,
        "no reference answer",
    )


def print_summary(
    results: list[dict[str, Any]],
    scorer_names: list[str],
) -> None:
    successful = [result for result in results if "error" not in result]
    summary = summarize(successful)

    print("\n" + "=" * 70)
    print(f"{len(results)} cases, " f"{len(results) - len(successful)} app errors")
    print("=" * 70)

    for name in scorer_names:
        if name not in summary:
            print(f"{name:<22}   n/a")
            continue

        failed = sum(
            1
            for result in successful
            if name in result["scores"] and scorer_failed(result["scores"][name])
        )

        suffix = f", failed={failed}" if failed else ""

        print(
            f"{name:<22}"
            f"{summary[name]['score']:>6.2f}"
            f"  (n={summary[name]['count']}{suffix})"
        )

    shown_scorers = [name for name in scorer_names if name in summary]

    if shown_scorers:
        print("\nBy tag, as mean (n):")
        print(f"{'tag':<14}" + "".join(f"{name:>20}" for name in shown_scorers))

        tags = sorted({tag for result in successful for tag in result["tags"]})

        for tag in tags:
            tag_results = [result for result in successful if tag in result["tags"]]
            tag_summary = summarize(tag_results)

            cells = [
                (
                    (
                        f"{tag_summary[name]['score']:.2f}"
                        f" ({tag_summary[name]['count']})"
                    )
                    if name in tag_summary
                    else "-"
                )
                for name in shown_scorers
            ]

            print(f"{tag:<14}" + "".join(f"{cell:>20}" for cell in cells))

    if successful:
        latencies = [result["latency"] for result in successful]
        print(
            f"\nLatency: mean {statistics.mean(latencies):.1f}s, "
            f"max {max(latencies):.1f}s"
        )

    problems: list[str] = []

    for result in results:
        if "error" in result:
            problems.append(f"  {result['id']}: " f"APP ERROR {result['error']}")
            continue

        for name, score in result["scores"].items():
            if scorer_failed(score):
                reason = score.get("error") or score.get("reason")
                problems.append(f"  {result['id']}: " f"{name} failed ({reason})")

    print("\nBroken cases " "(the app crashed or a scorer could not score):")
    print("\n".join(problems) if problems else "  none")

    print("\nFor why answers are wrong, " "run diagnose.py on the saved file.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the RAG app over the dataset, score it, "
            "and save the complete evaluation run."
        )
    )

    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help="configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="evaluation dataset",
    )
    parser.add_argument(
        "--tag",
        help="only run cases containing this tag",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="only run the first N selected cases",
    )
    parser.add_argument(
        "--scorers",
        default="all",
        help=(
            "'all' or a comma-separated list of scorers: " f"{', '.join(ALL_SCORERS)}"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="number of parallel workers (default: 2)",
    )
    parser.add_argument(
        "--judge-model",
        help="judge model; defaults to judge_model from config",
    )
    parser.add_argument(
        "--votes",
        type=int,
        default=judge.VOTES,
        help="number of judge votes for correctness",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output path for the saved run JSON",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="run and print results without saving a JSON artifact",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than 0.")

    if args.workers <= 0:
        raise SystemExit("--workers must be greater than 0.")

    if args.votes <= 0:
        raise SystemExit("--votes must be greater than 0.")


def select_dataset(
    dataset_path: Path,
    *,
    tag: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    dataset = load_dataset(dataset_path)

    if tag:
        dataset = [example for example in dataset if tag in example.get("tags", [])]

    if limit is not None:
        dataset = dataset[:limit]

    if not dataset:
        raise SystemExit("No cases to run " "(check --tag and --limit).")

    return dataset


def configure_judge(
    config: dict[str, Any],
    *,
    model: str | None,
    votes: int,
) -> str:
    selected_model = model or config.get("judge_model") or judge.MODEL

    judge.MODEL = selected_model
    judge.VOTES = votes

    return selected_model


def print_configuration_warning(
    config: dict[str, Any],
    *,
    judge_model: str,
    uses_judge: bool,
) -> None:
    if not uses_judge:
        return

    if config.get("model") != judge_model:
        return

    print(
        f"Warning: application and judge both use {judge_model}. "
        "Judge scores may be biased upward."
    )


def run_dataset(
    app: RagApp,
    dataset: list[dict[str, Any]],
    scorers: dict[str, Any],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def execute(example: dict[str, Any]) -> dict[str, Any]:
        return run_case(app, example, scorers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(
            pool.map(execute, dataset),
            start=1,
        ):
            results.append(result)

            score_text = " ".join(
                f"{name}={score['score']:.2f}"
                for name, score in result["scores"].items()
                if score.get("score") is not None
            )

            if "error" in result:
                status = "APP ERROR"
            else:
                status = score_text or "no scores"

            print(f"[{index}/{len(dataset)}] " f"{result['id']}  {status}")

    return results


def build_metadata(
    *,
    run_id: str,
    name: str,
    config: dict[str, Any],
    dataset_path: Path,
    dataset_size: int,
    duration: float,
    tag: str | None,
    limit: int | None,
    scorer_names: list[str],
    workers: int,
    judge_model: str | None,
    judge_votes: int | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "name": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(duration, 1),
        "config": config,
        "dataset": dataset_path.name,
        "dataset_sha": file_hash(dataset_path),
        "n_cases": dataset_size,
        "tag": tag,
        "limit": limit,
        "scorers": scorer_names,
        "workers": workers,
        "judge_model": judge_model,
        "judge_votes": judge_votes,
    }


def save_run(
    *,
    output_path: Path,
    metadata: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": metadata,
        "results": results,
    }

    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )

    output_path.write_text(
        scrub_home_paths(text),
        encoding="utf-8",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    validate_args(args)

    scorers = parse_scorers(args.scorers)
    scorer_names = list(scorers)
    uses_judge = bool(JUDGE_SCORERS.keys() & scorers.keys())

    config = load_config(args.config)
    dataset = select_dataset(
        args.dataset,
        tag=args.tag,
        limit=args.limit,
    )

    judge_model = configure_judge(
        config,
        model=args.judge_model,
        votes=args.votes,
    )

    print_configuration_warning(
        config,
        judge_model=judge_model,
        uses_judge=uses_judge,
    )

    run_id = uuid.uuid4().hex[:8]
    name = config.get("name", "run")

    app = RagApp(config)

    print(
        f"{name} [{run_id}]: "
        f"{len(app.chunks)} chunks, "
        f"{len(dataset)} cases, "
        f"{args.workers} worker(s), "
        f"scorers: {', '.join(scorer_names)}"
    )

    started = time.perf_counter()

    results = run_dataset(
        app,
        dataset,
        scorers,
        workers=args.workers,
    )

    duration = time.perf_counter() - started

    print_summary(
        results,
        scorer_names,
    )

    if args.no_save:
        return

    metadata = build_metadata(
        run_id=run_id,
        name=name,
        config=config,
        dataset_path=args.dataset,
        dataset_size=len(dataset),
        duration=duration,
        tag=args.tag,
        limit=args.limit,
        scorer_names=scorer_names,
        workers=args.workers,
        judge_model=judge_model if uses_judge else None,
        judge_votes=args.votes if uses_judge else None,
    )

    safe_name = re.sub(r"[^\w.-]+", "_", name)
    output_path = args.output or RESULTS_DIR / f"{safe_name}_{run_id}.json"

    save_run(
        output_path=output_path,
        metadata=metadata,
        results=results,
    )

    print(f"\nSaved: {output_path}")
    print(f"Run id: {run_id}")


if __name__ == "__main__":
    main()
