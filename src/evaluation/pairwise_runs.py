import argparse
import json
from pathlib import Path

from scipy.stats import binomtest

from evaluation import llm_judge_scorers as judge
from evaluation.compare import load_run
from evaluation.pairwise import compare_pair
from evaluation.trace import find_run_by_id, find_run_file


def resolve(value: str) -> Path:
    try:
        return find_run_file(value)
    except SystemExit:
        return find_run_by_id(value)


def usable(run: dict) -> dict[str, dict]:
    return {
        r["id"]: r
        for r in run["results"]
        if "error" not in r and r.get("answer") and r.get("expected") is not None
    }


def label(run: dict) -> str:
    meta = run["meta"]
    return f"{meta.get('name', 'run')} [{meta.get('run_id', '?')}]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two saved runs with a position-swapped pairwise judge."
    )
    parser.add_argument("--a", required=True, help="run A: file, name in runs/, or run_id")
    parser.add_argument("--b", required=True, help="run B: file, name in runs/, or run_id")
    parser.add_argument("--tag", help="only compare cases with this tag")
    parser.add_argument("--limit", type=int, help="only compare the first N shared cases")
    parser.add_argument("--judge-model", help="default: judge_model from run A's config")
    parser.add_argument("--output", type=Path, help="save every verdict as JSON")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than 0.")

    run_a, run_b = load_run(resolve(args.a)), load_run(resolve(args.b))
    answers_a, answers_b = usable(run_a), usable(run_b)

    ids = [
        i
        for i in answers_a
        if i in answers_b
        and (not args.tag or args.tag in answers_a[i].get("tags", []))
    ]
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        raise SystemExit("No shared answerable cases to compare (check --tag and the runs).")

    cfg_a, cfg_b = run_a["meta"].get("config", {}), run_b["meta"].get("config", {})
    model = args.judge_model or cfg_a.get("judge_model") or judge.MODEL
    judge.MODEL = model

    print(f"A: {label(run_a)}\nB: {label(run_b)}\nJudge: {model}, {len(ids)} cases\n")
    if model in (cfg_a.get("model"), cfg_b.get("model")):
        print(f"Warning: the judge ({model}) is also one of the models being judged.\n")
    if run_a["meta"].get("dataset_sha") != run_b["meta"].get("dataset_sha"):
        print("Warning: the two runs used different dataset files.\n")

    verdicts, errors = [], []
    for index, case_id in enumerate(ids, start=1):
        a, b = answers_a[case_id], answers_b[case_id]
        try:
            verdict = compare_pair(a["question"], a["expected"], a["answer"], b["answer"])
        except Exception as exc:
            errors.append(case_id)
            print(f"[{index}/{len(ids)}] {case_id}  ERROR {type(exc).__name__}: {exc}")
            continue
        verdicts.append({"id": case_id, **verdict})
        shown = {"1": "A wins", "2": "B wins", "tie": f"tie ({verdict['kind']})"}
        print(f"[{index}/{len(ids)}] {case_id}  {shown[verdict['winner']]}")

    a_wins = [v["id"] for v in verdicts if v["winner"] == "1"]
    b_wins = [v["id"] for v in verdicts if v["winner"] == "2"]
    same = [v["id"] for v in verdicts if v["kind"] == "identical"]
    biased = [v["id"] for v in verdicts if v["kind"] == "position-biased"]

    print("\n" + "=" * 70)
    print(f"{len(verdicts)} judged, {len(errors)} judge errors")
    print("=" * 70)
    print(f"A wins                  {len(a_wins)}")
    print(f"B wins                  {len(b_wins)}")
    print(f"tie, identical answers  {len(same)}")
    print(f"tie, position-biased    {len(biased)}")

    decided = len(a_wins) + len(b_wins)
    if decided:
        p = binomtest(len(b_wins), decided, 0.5).pvalue
        print(f"\nOf {decided} decided cases, B won {len(b_wins)} (sign test p = {p:.3f}).")
        print("A small p means one run is really better; a large p means you cannot tell.")
    asked = decided + len(biased)
    if biased and len(biased) > 0.3 * asked:
        print(f"\nMany position-biased ties ({len(biased)} of the {asked} pairs the judge saw): "
              "the judge is swayed by order, so trust the wins less.")
    if a_wins:
        print(f"\nA better on: {', '.join(a_wins)}")
    if b_wins:
        print(f"B better on: {', '.join(b_wins)}")
    if errors:
        print(f"Judge failed on: {', '.join(errors)}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"a": label(run_a), "b": label(run_b), "judge": model, "verdicts": verdicts, "errors": errors},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()