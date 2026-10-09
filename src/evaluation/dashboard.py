import difflib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from evaluation.code_evaluation import DATASET_PATH, load_dataset, summarize
from evaluation.compare import changed_cases, compare
from evaluation.diagnose import diagnose


st.set_page_config(page_title="Eval dashboard", layout="wide")


st.sidebar.header("Settings")
folder = st.sidebar.text_input("Runs folder", "runs")

runs = {}

for path in Path(folder).glob("*.json"):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        continue

    if (
        isinstance(data, dict)
        and isinstance(data.get("meta"), dict)
        and isinstance(data.get("results"), list)
    ):
        runs[path.name] = data


if not runs:
    st.info(
        f"No valid runs found in '{folder}'. "
        f"Save one with: uv run run_eval.py --output {folder}/v1.json"
    )
    st.stop()


names = sorted(
    runs,
    key=lambda name: str(runs[name]["meta"].get("timestamp") or ""),
    reverse=True,
)


name = st.sidebar.selectbox("Run", names)
run = runs[name]
meta = run["meta"]
results = run["results"]

ok = [
    result
    for result in results
    if "error" not in result
]


try:
    dataset = {
        row["id"]: row
        for row in load_dataset(DATASET_PATH)
    }
except (OSError, ValueError, KeyError, TypeError) as exc:
    dataset = {}
    st.warning(f"Could not load the evaluation dataset: {exc}")


st.title(
    f"{meta.get('name', name)} "
    f"[{meta.get('run_id', 'unknown')}]"
)

overview_tab, compare_tab, cases_tab = st.tabs(
    ["Overview", "Compare", "Cases"]
)


with overview_tab:
    timestamp = meta.get("timestamp", "Unknown time")
    n_cases = meta.get("n_cases", len(results))
    judge_model = meta.get("judge_model") or "not used"

    st.caption(
        f"{timestamp} | {n_cases} cases | judge: {judge_model}"
    )

    failed = len(results) - len(ok)

    if failed:
        st.error(f"{failed} cases crashed the app.")

    if not ok:
        st.warning("This run has no successful cases.")
    else:
        summaries = summarize(ok)

        columns = st.columns(max(1, min(4, len(summaries))))

        for i, (scorer, summary) in enumerate(summaries.items()):
            score = summary.get("score")

            columns[i % len(columns)].metric(
                scorer,
                f"{score:.2f}" if score is not None else "N/A",
                help=f"{summary.get('count', 0)} cases",
            )

        latencies = [
            result["latency"]
            for result in ok
            if isinstance(result.get("latency"), (int, float))
            and not isinstance(result.get("latency"), bool)
        ]

        if latencies:
            mean_latency = sum(latencies) / len(latencies)
            st.write(
                f"Latency: mean {mean_latency:.1f}s, "
                f"slowest {max(latencies):.1f}s"
            )
        else:
            st.write("Latency data is unavailable.")

        st.subheader("By tag")

        tags = sorted({
            tag
            for result in ok
            for tag in result.get("tags", [])
        })

        rows = {}

        for tag in tags:
            tagged = [
                result
                for result in ok
                if tag in result.get("tags", [])
            ]

            rows[f"{tag} ({len(tagged)})"] = {
                scorer: values["score"]
                for scorer, values in summarize(tagged).items()
            }

        if rows:
            st.dataframe(
                pd.DataFrame(rows).T.round(2),
                use_container_width=True,
            )
        else:
            st.info("No tags found in this run.")


with compare_tab:
    if len(names) < 2:
        st.info("You need at least two run files to compare.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            a = st.selectbox(
                "A (baseline)",
                names,
                index=1,
                key="compare_a",
            )

        with col_b:
            b = st.selectbox(
                "B (change)",
                names,
                index=0,
                key="compare_b",
            )

        run_a = runs[a]
        run_b = runs[b]

        meta_a = run_a["meta"]
        meta_b = run_b["meta"]

        config_a = meta_a.get("config", {})
        config_b = meta_b.get("config", {})

        changed_keys = [
            key
            for key in sorted(set(config_a) | set(config_b))
            if key != "name"
            and config_a.get(key) != config_b.get(key)
        ]

        st.subheader("Configuration differences")

        if changed_keys:
            diff_lines = []

            for key in changed_keys:
                before = config_a.get(key)
                after = config_b.get(key)

                if (
                    isinstance(before, str)
                    and isinstance(after, str)
                    and ("\n" in before or "\n" in after)
                ):
                    diff_lines.append(f"{key}:")

                    diff_lines.extend(
                        difflib.unified_diff(
                            before.splitlines(),
                            after.splitlines(),
                            fromfile="A",
                            tofile="B",
                            lineterm="",
                        )
                    )
                else:
                    diff_lines.append(
                        f"{key}: {before!r} -> {after!r}"
                    )

            st.code("\n".join(diff_lines), language="text")
        else:
            st.success("No configuration differences found.")

        warnings = []

        if meta_a.get("dataset_sha") != meta_b.get("dataset_sha"):
            warnings.append(
                "The dataset files differ. "
                "The evaluated cases may not be the same."
            )

        if (
            meta_a.get("tag") != meta_b.get("tag")
            or meta_a.get("limit") != meta_b.get("limit")
        ):
            warnings.append(
                "The runs used different --tag or --limit filters."
            )

        if meta_a.get("judge_model") != meta_b.get("judge_model"):
            warnings.append(
                "The judge models differ. "
                "Judge scores may not be comparable."
            )

        if meta_a.get("judge_votes") != meta_b.get("judge_votes"):
            warnings.append(
                "The runs used different judge vote counts."
            )

        if warnings:
            st.subheader("Warnings")

            for warning in warnings:
                st.warning(warning)

        st.subheader("Score comparison")

        try:
            comparison = compare(run_a, run_b)

            if comparison.empty:
                st.info(
                    "No scorer has at least two cases scored in both runs."
                )
            else:
                st.caption(
                    "Diff = B - A. The confidence interval indicates "
                    "the uncertainty in the mean paired score difference."
                )

                st.dataframe(
                    comparison,
                    hide_index=True,
                    use_container_width=True,
                )

        except (KeyError, TypeError, ValueError) as exc:
            st.error(f"Could not compare these runs: {exc}")

        st.subheader("Cases that changed")

        try:
            changes = changed_cases(run_a, run_b)

            if not changes:
                st.info("No score changes found for shared cases.")
            else:
                for scorer, (moved, total) in changes.items():
                    st.write(
                        f"**{scorer}**: {len(moved)} of {total} "
                        "shared cases changed"
                    )

                    changed_df = pd.DataFrame(
                        moved,
                        columns=["Case", "A", "B"],
                    )

                    changed_df["Diff"] = (
                        changed_df["B"] - changed_df["A"]
                    )

                    st.dataframe(
                        changed_df.round(3),
                        hide_index=True,
                        use_container_width=True,
                    )

        except (KeyError, TypeError, ValueError) as exc:
            st.error(f"Could not list changed cases: {exc}")


with cases_tab:
    scorers = meta.get("scorers", [])
    judged = "correctness" in scorers

    if not judged:
        st.info(
            "This run has no correctness judge scores. "
            "Diagnosis may be unavailable."
        )

    rows = []

    for result in results:
        case_id = result.get("id", "unknown")
        scores = result.get("scores", {})

        def get_score(scorer):
            score_data = scores.get(scorer)
            if isinstance(score_data, dict):
                return score_data.get("score")
            return None

        diagnosis = "-"

        if "error" in result:
            diagnosis = "error"
        elif judged and case_id in dataset:
            try:
                diagnosis = diagnose(dataset[case_id], result)
            except (KeyError, TypeError, ValueError):
                diagnosis = "diagnosis error"

        tags = result.get("tags", [])

        rows.append({
            "id": case_id,
            "diagnosis": diagnosis,
            "correctness": get_score("correctness"),
            "faithfulness": get_score("faithfulness"),
            "tags": ", ".join(tags) if isinstance(tags, list) else "",
        })

    table = pd.DataFrame(rows)

    if table.empty:
        st.info("This run has no cases.")
    else:
        labels = sorted(
            table["diagnosis"].dropna().unique().tolist()
        )

        default_labels = [
            label
            for label in labels
            if label not in ("ok", "-")
        ]

        selected = st.multiselect(
            "Diagnosis",
            labels,
            default=default_labels,
        )

        filtered = (
            table[table["diagnosis"].isin(selected)]
            if selected
            else table.iloc[0:0]
        )

        st.dataframe(
            filtered,
            hide_index=True,
            use_container_width=True,
        )

        if not filtered.empty:
            case_ids = filtered["id"].tolist()

            selected_id = st.selectbox(
                "Open a case",
                case_ids,
                key="open_case",
            )

            selected_result = next(
                (
                    item
                    for item in results
                    if item.get("id") == selected_id
                ),
                None,
            )

            if selected_result is not None:
                st.divider()
                st.subheader(f"Case: {selected_id}")

                if "error" in selected_result:
                    st.error(selected_result["error"])
                else:
                    st.markdown(
                        f"**Question**\n\n"
                        f"{selected_result.get('question') or '(none)'}"
                    )

                    st.markdown(
                        f"**Reference**\n\n"
                        f"{selected_result.get('expected') or '(none, the handbook does not say)'}"
                    )

                    st.markdown(
                        f"**Answer**\n\n"
                        f"{selected_result.get('answer') or '(none)'}"
                    )

                    st.subheader("Scores and feedback")

                    scores = selected_result.get("scores", {})

                    for scorer, score_data in scores.items():
                        if not isinstance(score_data, dict):
                            continue

                        score = score_data.get("score")

                        if score is not None:
                            st.write(
                                f"`{scorer}` {score:.2f}  "
                                f"{score_data.get('reason', '')}"
                            )

                        for claim in score_data.get("unsupported", []):
                            st.write(f"- Unsupported: {claim}")

                    with st.expander("Retrieved chunks"):
                        contexts = selected_result.get("contexts", [])

                        if contexts:
                            for i, chunk in enumerate(contexts, start=1):
                                st.text(f"Chunk {i}\n{chunk}")
                        else:
                            st.info("No retrieved chunks available.")