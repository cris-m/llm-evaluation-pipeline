import argparse
import glob
import json
import os
import sys
from pathlib import Path

from evaluation.app import chunk_document, load_config
from evaluation.code_scorers import _has_token, _normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs", "handbook")
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "dataset.jsonl")

normalize = _normalize


def resolve(path, what):
    path = str(path)
    for candidate in (path, os.path.join(BASE_DIR, path)):
        if os.path.exists(candidate):
            return candidate
    raise SystemExit(
        f"{what} not found: '{path}'\n" f"  looked in: {os.getcwd()} and {BASE_DIR}"
    )


def doc_paths(docs_dir=DOCS_DIR):
    paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
    if not paths:
        raise SystemExit(f"No .txt files found in '{docs_dir}/'")
    return paths


def load_docs(docs_dir=DOCS_DIR):
    texts = []
    for path in doc_paths(docs_dir):
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return normalize(" ".join(texts))


def load_dataset(dataset_path=DATASET_PATH):
    if not Path(dataset_path).exists():
        raise SystemExit(f"Dataset not found: '{dataset_path}'")

    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON on line {number}: {e}")
    return rows


def dedupe(rows):
    seen, kept, removed = set(), [], []
    for row in rows:
        key = normalize(row["question"])
        if key in seen:
            removed.append(row["id"])
        else:
            seen.add(key)
            kept.append(row)
    return kept, removed


def validate(rows, docs):
    errors = []
    required = {"id", "question", "expected", "must_contain", "evidence", "tags"}

    broken = set()
    for index, row in enumerate(rows, start=1):
        missing = required - row.keys()
        if missing:
            errors.append(
                f'{row.get("id", f"line {index}")}: missing fields {sorted(missing)}'
            )
            broken.add(index)
    rows = [r for i, r in enumerate(rows, start=1) if i not in broken]

    by_id, by_question = {}, {}
    for row in rows:
        by_id.setdefault(row["id"], []).append(row["id"])
        by_question.setdefault(normalize(str(row["question"])), []).append(row["id"])
    for rid, ids in by_id.items():
        if len(ids) > 1:
            errors.append(f"{rid}: duplicate id (used {len(ids)} times)")
    for question, ids in by_question.items():
        if len(ids) > 1:
            errors.append(f"{', '.join(ids)}: duplicate question: {question!r}")

    for row in rows:
        rid = row["id"]

        if not isinstance(row["question"], str):
            errors.append(f"{rid}: question must be a string")
        if row["expected"] is not None and not isinstance(row["expected"], str):
            errors.append(f"{rid}: expected must be a string or null")
        bad_type = False
        for field in ("must_contain", "evidence", "tags"):
            if not isinstance(row[field], list):
                errors.append(f"{rid}: {field} must be a list")
                bad_type = True
        if bad_type or not isinstance(row["question"], str):
            continue

        if ("unanswerable" in row["tags"]) != (row["expected"] is None):
            errors.append(
                f"{rid}: the 'unanswerable' tag and expected=null must go together"
            )

        if row["expected"] is None:
            if row["evidence"]:
                errors.append(f"{rid}: unanswerable cases need empty evidence")
            if row["must_contain"]:
                errors.append(f"{rid}: unanswerable cases need empty must_contain")
        elif not row["evidence"]:
            errors.append(f"{rid}: answerable cases need evidence")

        for quote in row["evidence"]:
            if normalize(str(quote)) not in docs:
                errors.append(f"{rid}: evidence not found in docs: {quote!r}")

        if isinstance(row["expected"], str):
            expected = normalize(row["expected"])
            for value in row["must_contain"]:
                if not _has_token(str(value), expected):
                    errors.append(
                        f'{rid}: must_contain "{value}" is not a whole token in expected'
                    )

    return errors


def check_evidence_is_retrievable(rows, docs_dir=DOCS_DIR, config_path=None):
    try:
        cfg = load_config(config_path) if config_path else load_config()
    except FileNotFoundError as e:
        raise SystemExit(f"Config not found: {e.filename}")
    except KeyError as e:
        raise SystemExit(f"Config is missing the key {e}")
    chunks = []
    for path in doc_paths(docs_dir):
        with open(path, "r", encoding="utf-8") as f:
            for c in chunk_document(
                f.read(), path, cfg["chunk_size"], cfg["overlap_sentences"]
            ):
                chunks.append(normalize(c["text"]))

    errors = []
    for row in rows:
        for quote in row.get("evidence", []):
            if not any(normalize(str(quote)) in c for c in chunks):
                errors.append(
                    f'{row["id"]}: evidence is split across chunks (or sections): {quote!r}'
                )
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate a test dataset against the handbook."
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_PATH,
        help=f"dataset .jsonl (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--docs",
        default=DOCS_DIR,
        help=f"folder with the .txt docs (default: {DOCS_DIR})",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="config used for chunk_size/overlap (default: configs/v1.yaml)",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="drop duplicate questions and save a cleaned copy next to the dataset",
    )
    args = parser.parse_args()

    try:
        args.dataset = resolve(args.dataset, "Dataset")
    except SystemExit as e:
        raise SystemExit(
            f"{e}\n  Create it by hand, or generate a draft first: uv run make_dataset.py"
        )
    args.docs = resolve(args.docs, "Docs folder")
    if args.config:
        args.config = resolve(args.config, "Config")

    print(f"Dataset: {args.dataset}")
    print(f"Docs:    {args.docs}")
    if args.config:
        print(f"Config:  {args.config}")

    docs = load_docs(args.docs)
    rows = load_dataset(args.dataset)

    if args.dedupe:
        rows, removed = dedupe(rows)
        out = Path(args.dataset).with_name(Path(args.dataset).stem + "_deduped.jsonl")
        out.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        print(
            f"Removed {len(removed)} duplicates ({', '.join(removed) or 'none'}). Saved: {out}"
        )

    if not rows:
        raise SystemExit(f"The dataset is empty: {args.dataset}")

    errors = validate(rows, docs)
    errors += check_evidence_is_retrievable(
        [r for r in rows if isinstance(r.get("evidence"), list) and "id" in r],
        args.docs,
        args.config,
    )

    if errors:
        print(f"\n{len(errors)} problems found:")
        for e in errors:
            print(f"  - {e}")
        if any("duplicate question" in e for e in errors) and not args.dedupe:
            print(
                "\nTip: rerun with --dedupe to drop duplicate questions automatically."
            )
        sys.exit(1)

    tags = {t for r in rows for t in r["tags"]}
    print(f"{len(rows)} cases OK, {len(tags)} tags")


if __name__ == "__main__":
    main()
