import argparse
import difflib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_run(path: Path) -> dict:
    if not Path(path).exists():
        raise SystemExit(f"Run file not found: {path}")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise SystemExit(f"{path}: not a JSON run file (was it saved by run_eval.py?).")
    if not isinstance(data, dict) or "meta" not in data or "results" not in data:
        raise SystemExit(
            f"{path}: not a run saved by run_eval.py (no 'meta' and 'results')."
        )
    return data


def scores_by_scorer(run: dict) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for result in run["results"]:
        if "error" in result:
            continue
        for name, scored in result["scores"].items():
            if scored.get("score") is not None:
                table.setdefault(name, {})[result["id"]] = scored["score"]
    return table


def compare(
    run_a: dict, run_b: dict, n_boot: int = 5000, level: float = 95.0, seed: int = 0
) -> pd.DataFrame:
    a, b = scores_by_scorer(run_a), scores_by_scorer(run_b)
    rng = np.random.default_rng(seed)
    tail = (100 - level) / 2
    rows = []

    for name in (n for n in a if n in b):
        paired = sorted(set(a[name]) & set(b[name]))
        if len(paired) < 2:
            continue
        x = np.array([a[name][i] for i in paired])
        y = np.array([b[name][i] for i in paired])
        diff = y - x

        picks = rng.integers(0, len(diff), size=(n_boot, len(diff)))
        means = diff[picks].mean(axis=1)
        low, high = np.percentile(means, [tail, 100 - tail])

        if np.all(diff == 0):
            verdict = "identical"
        elif low > 0:
            verdict = "B better"
        elif high < 0:
            verdict = "A better"
        else:
            verdict = "no clear difference"

        rows.append(
            {
                "scorer": name,
                "A": x.mean(),
                "B": y.mean(),
                "diff": diff.mean(),
                "ci_low": low,
                "ci_high": high,
                "verdict": verdict,
                "n": len(paired),
                "unpaired": len(set(a[name]) ^ set(b[name])),
            }
        )
    return pd.DataFrame(rows).round(3)


def changed_cases(run_a: dict, run_b: dict) -> dict:
    a, b = scores_by_scorer(run_a), scores_by_scorer(run_b)
    changes = {}
    for name in (n for n in a if n in b):
        paired = sorted(set(a[name]) & set(b[name]))
        moved = [
            (i, a[name][i], b[name][i])
            for i in paired
            if abs(b[name][i] - a[name][i]) > 1e-9
        ]
        if moved:
            moved.sort(key=lambda row: -abs(row[2] - row[1]))
            changes[name] = (moved, len(paired))
    return changes


def print_changes(changes: dict, limit: int) -> None:
    print("\nCases that changed (A -> B):")
    if not changes:
        print("  none")
        return
    for name, (moved, total) in changes.items():
        print(f"\n{name}: {len(moved)} of {total} cases")
        for case_id, x, y in moved[:limit]:
            print(f"  {case_id}  {x:.2f} -> {y:.2f}  ({y - x:+.2f})")
        if len(moved) > limit:
            print(f"  ... and {len(moved) - limit} more")


def short(value, width: int = 60) -> str:
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 3] + "..."


def print_header(run_a: dict, run_b: dict) -> None:
    ma, mb = run_a["meta"], run_b["meta"]
    print(f"A: {ma['name']} [{ma['run_id']}]  {ma['n_cases']} cases")
    print(f"B: {mb['name']} [{mb['run_id']}]  {mb['n_cases']} cases")

    changed = [
        k
        for k in sorted(set(ma["config"]) | set(mb["config"]))
        if k != "name" and ma["config"].get(k) != mb["config"].get(k)
    ]
    print("\nWhat changed in the config:")
    for k in changed:
        before, after = ma["config"].get(k), mb["config"].get(k)
        if (
            isinstance(before, str)
            and isinstance(after, str)
            and "\n" in (before + after)
        ):
            print(f"  {k}:")
            for line in difflib.unified_diff(
                before.splitlines(), after.splitlines(), lineterm="", n=0
            ):
                if line[:1] in "+-" and line[:3] not in ("+++", "---"):
                    print(f"    {line}")
        else:
            print(f"  {k}: {short(before)}  ->  {short(after)}")
    if not changed:
        print("  nothing (same config: this shows you the run-to-run noise)")

    warnings = []
    if ma["dataset_sha"] != mb["dataset_sha"]:
        warnings.append("the dataset file is different, so the cases are not the same")
    if ma.get("tag") != mb.get("tag") or ma.get("limit") != mb.get("limit"):
        warnings.append("the runs used different --tag/--limit filters")
    if ma.get("judge_model") != mb.get("judge_model"):
        warnings.append(
            f"different judges ({ma.get('judge_model')} vs {mb.get('judge_model')}): judge scores are not comparable"
        )
    if ma.get("judge_votes") != mb.get("judge_votes"):
        warnings.append("different --votes")
    for w in warnings:
        print(f"\nWarning: {w}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Is run B really different from run A?"
    )
    parser.add_argument(
        "run_a", type=Path, help="run JSON saved by run_eval.py (the baseline)"
    )
    parser.add_argument(
        "run_b", type=Path, help="run JSON saved by run_eval.py (the change)"
    )
    parser.add_argument("--boots", type=int, default=5000, help="bootstrap resamples")
    parser.add_argument(
        "--level", type=float, default=95.0, help="confidence level in percent"
    )
    parser.add_argument(
        "--max-changes", type=int, default=10, help="changed cases to list per scorer"
    )
    args = parser.parse_args()

    run_a, run_b = load_run(args.run_a), load_run(args.run_b)
    print_header(run_a, run_b)

    table = compare(run_a, run_b, args.boots, args.level)
    if table.empty:
        raise SystemExit("\nNo scorer has at least 2 cases scored in both runs.")

    print(
        f"\n{args.level:g}% interval for diff = B - A (0 inside the interval means no clear difference):\n"
    )
    print(table.to_string(index=False))
    print(
        "\nn = cases compared. unpaired = cases scored in only one of the two runs (left out)."
    )
    print_changes(changed_cases(run_a, run_b), args.max_changes)


if __name__ == "__main__":
    main()
