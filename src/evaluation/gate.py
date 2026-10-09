import argparse
import math
import sys
from pathlib import Path

import yaml

from evaluation.code_evaluation import summarize
from evaluation.compare import load_run

FLOORS_PATH = Path(__file__).resolve().parent / "configs" / "gate.yaml"


def load_floors(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"No floors file at {path}. Write one with: uv run gate.py <good run>.json --write-floors"
        )
    floors = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not floors:
        raise SystemExit(f"{path} is empty.")
    return floors


def write_floors(means: dict, margin: float, path: Path) -> None:
    floors = {
        name: math.floor(round((s["score"] - margin) * 100, 6)) / 100
        for name, s in means.items()
        if name != "latency_ok"
    }
    path.write_text(yaml.safe_dump(floors, sort_keys=False), encoding="utf-8")
    print(f"Wrote {path.name}:")
    for name, floor in floors.items():
        print(f"  {name:<22}{floor:.2f}   (this run: {means[name]['score']:.3f})")


def check(means: dict, floors: dict) -> list[str]:
    failed = []
    print(f"{'scorer':<22}{'mean':>7}{'floor':>7}")
    for name, floor in floors.items():
        if name not in means:
            print(f"{name:<22}{'none':>7}{floor:>7.2f}  FAIL (not scored in this run)")
            failed.append(name)
            continue
        mean = means[name]["score"]
        passed = round(mean, 6) >= floor
        print(f"{name:<22}{mean:>7.3f}{floor:>7.2f}  {'ok' if passed else 'FAIL'}")
        if not passed:
            failed.append(name)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail when a run scores below its floors."
    )
    parser.add_argument("run", type=Path, help="run saved by run_eval.py")
    parser.add_argument(
        "--floors", type=Path, default=FLOORS_PATH, help="default: configs/gate.yaml"
    )
    parser.add_argument(
        "--write-floors",
        action="store_true",
        help="write the floors from this run instead of checking it",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="room below the run's averages when writing floors",
    )
    args = parser.parse_args()

    run = load_run(args.run)
    ok = [r for r in run["results"] if "error" not in r]
    means = summarize(ok)

    if args.write_floors:
        write_floors(means, args.margin, args.floors)
        return

    failed = check(means, load_floors(args.floors))

    crashed = len(run["results"]) - len(ok)
    print(f"\napp errors: {crashed}")
    if crashed:
        failed.append("app errors")

    if failed:
        print(f"GATE FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("Gate passed.")


if __name__ == "__main__":
    main()
