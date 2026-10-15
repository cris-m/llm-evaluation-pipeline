import argparse
import json
from pathlib import Path

from evaluation.code_evaluation import DATASET_PATH, RESULTS_DIR, load_dataset
from evaluation.code_scorers import _normalize
from evaluation.compare import load_run



RUN_DIRS = (Path(__file__).resolve().parent / "runs", RESULTS_DIR)
FOLDERS = " and ".join(f"{d.name}/" for d in RUN_DIRS)


def find_run_file(value: str) -> Path:
    candidates = [Path(value)]
    for folder in RUN_DIRS:
        candidates += [folder / value, folder / f"{value}.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(f"Run file not found: '{value}' (also looked in {FOLDERS}).")


def find_run_by_id(run_id: str) -> Path:
    for folder in RUN_DIRS:
        for candidate in sorted(folder.glob("*.json")):
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            meta = data.get("meta") if isinstance(data, dict) else None
            if isinstance(meta, dict) and meta.get("run_id") == run_id:
                return candidate
    raise SystemExit(f"No run with run_id '{run_id}' in {FOLDERS}.")


def problem_scorers(result: dict, ignore: set[str] = frozenset()) -> list[str]:
    if "error" in result:
        return ["APP ERROR"]
    problems = []
    for name, scored in result["scores"].items():
        if name in ignore or not isinstance(scored, dict):
            continue
        if "error" in scored:
            problems.append(f"{name}(ERROR)")
        elif scored.get("score") is not None and scored["score"] < 1:
            problems.append(f"{name}={scored['score']:.2f}")
    return problems


def list_cases(results: list[dict], ignore: set[str]) -> None:
    shown_ignored = f" (ignoring {', '.join(sorted(ignore))})" if ignore else ""
    print(f"\nCases to look at{shown_ignored}:")
    shown = 0
    for result in results:
        problems = problem_scorers(result, ignore)
        if problems:
            shown += 1
            print(f"  {result['id']}: {', '.join(problems)}")
    if not shown:
        print("  none")
    print("\nOpen one with: uv run trace.py --run <run> --case <case id>")


def load_examples(path: Path) -> dict[str, dict]:
    try:
        return {row["id"]: row for row in load_dataset(path)}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def show_case(result: dict, example: dict | None, meta: dict) -> None:
    print("=" * 70)
    print(f"{meta.get('name', 'run')} [{meta.get('run_id', '?')}]   case {result['id']}")
    print("=" * 70)

    if "error" in result:
        print(f"\nAPP ERROR: {result['error']}")
        return

    print(f"\nTags:      {', '.join(result.get('tags', [])) or '-'}")
    if isinstance(result.get("latency"), (int, float)):
        print(f"Latency:   {result['latency']:.2f}s")
    print(f"Sources:   {', '.join(result.get('sources', [])) or '-'}")

    print(f"\nQuestion:\n{result['question']}")
    print(f"\nReference:\n{result.get('expected') or '(none, the handbook does not say)'}")
    print(f"\nAnswer:\n{result.get('answer') or '(none)'}")

    evidence = [_normalize(str(q)) for q in (example or {}).get("evidence", [])]
    contexts = result.get("contexts", [])

    print(f"\nRetrieved chunks ({len(contexts)}):")
    if not contexts:
        print("  (none: retrieval returned nothing, so the app abstained)")
    for i, chunk in enumerate(contexts, start=1):
        hit = any(q in _normalize(chunk) for q in evidence)
        print(f"\n[chunk {i}]{'  <- contains evidence' if hit else ''}\n{chunk}")
    if evidence and contexts and not any(
        q in _normalize(c) for q in evidence for c in contexts
    ):
        print("\nNo retrieved chunk contains the evidence: this is a retrieval miss.")

    print("\nScores:")
    for name, scored in result["scores"].items():
        if not isinstance(scored, dict):
            continue
        score = scored.get("score")
        text = f"{score:.2f}" if score is not None else "n/a"
        detail = scored.get("error") or scored.get("reason") or ""
        print(f"  {name:<22}{text:>6}  {detail}")
        if scored.get("votes"):
            print(f"{'':<30}votes: {scored['votes']}")
        if scored.get("missing"):
            print(f"{'':<30}missing: {scored['missing']}")
        if scored.get("rank") is not None:
            print(f"{'':<30}evidence first at rank {scored['rank']}")
        for claim in scored.get("unsupported", []):
            print(f"{'':<30}unsupported: {claim}")

    if example and "correctness" in meta.get("scorers", []):
        try:
            from evaluation.diagnose import diagnose

            print(f"\nDiagnosis: {diagnose(example, result)}")
        except Exception as exc:
            print(f"\nDiagnosis unavailable ({type(exc).__name__}: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show one case from a saved run: answer, chunks, scores, diagnosis."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run",
        help="run file or its name in the runs folder, e.g. runs/bad.json or v3",
    )
    source.add_argument("--run-id", help="the run_id printed when the run was saved")
    parser.add_argument(
        "--case", help="case id to open; leave out to list the cases to look at"
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        metavar="SCORER",
        help="scorers left out of the case list, e.g. --ignore latency_ok (default: none)",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    args = parser.parse_args()

    path = find_run_file(args.run) if args.run else find_run_by_id(args.run_id)
    run = load_run(path)
    meta, results = run["meta"], run["results"]

    if not args.case:
        print(f"{meta.get('name', 'run')} [{meta.get('run_id', '?')}]: {len(results)} cases")
        list_cases(results, set(args.ignore))
        return

    result = next((r for r in results if r.get("id") == args.case), None)
    if result is None:
        ids = ", ".join(str(r.get("id")) for r in results[:15])
        raise SystemExit(f"Case '{args.case}' is not in this run. First ids: {ids} ...")

    show_case(result, load_examples(args.dataset).get(args.case), meta)


if __name__ == "__main__":
    main()