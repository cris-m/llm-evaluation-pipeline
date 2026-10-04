import glob
import json
import re
from pathlib import Path


DOCS_DIR = "docs/handbook"
DATASET_PATH = "datasets/dataset.jsonl"


def normalize(text):
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_docs():
    """Load and combine all documentation files."""
    paths = glob.glob(f"{DOCS_DIR}/*.txt")

    if not paths:
        raise FileNotFoundError(
            f"No .txt files found in '{DOCS_DIR}/'"
        )

    documents = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            documents.append(f.read())

    return normalize(" ".join(documents))


def load_dataset():
    """Load JSONL dataset."""
    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found: '{DATASET_PATH}'"
        )

    rows = []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise AssertionError(
                    f"Invalid JSON on line {line_number}: {e}"
                )

    return rows


def validate(rows, docs):
    """Validate the dataset against the documentation."""

    required_fields = {
        "id",
        "question",
        "expected",
        "must_contain",
        "evidence",
        "tags",
    }

    for row in rows:
        missing = required_fields - row.keys()

        assert not missing, (
            f'{row.get("id", "<unknown>")}: '
            f"missing fields: {sorted(missing)}"
        )

    ids = [row["id"] for row in rows]

    assert len(ids) == len(set(ids)), (
        "duplicate ids: "
        + ", ".join(
            id_ for id_ in set(ids)
            if ids.count(id_) > 1
        )
    )

    questions = [
        normalize(row["question"])
        for row in rows
    ]

    assert len(questions) == len(set(questions)), (
        "duplicate questions"
    )

    for row in rows:
        row_id = row["id"]

        # Basic type checks
        assert isinstance(row["question"], str), (
            f"{row_id}: question must be a string"
        )

        assert (
            row["expected"] is None
            or isinstance(row["expected"], str)
        ), (
            f"{row_id}: expected must be a string or null"
        )

        assert isinstance(row["must_contain"], list), (
            f"{row_id}: must_contain must be a list"
        )

        assert isinstance(row["evidence"], list), (
            f"{row_id}: evidence must be a list"
        )

        assert isinstance(row["tags"], list), (
            f"{row_id}: tags must be a list"
        )

        if row["expected"] is None:
            assert not row["evidence"], (
                f"{row_id}: unanswerable cases need empty evidence"
            )

            assert not row["must_contain"], (
                f"{row_id}: unanswerable cases need empty must_contain"
            )

        if row["expected"] is not None:
            assert row["evidence"], (
                f"{row_id}: answerable cases need evidence"
            )

        for evidence in row["evidence"]:
            normalized_evidence = normalize(evidence)

            assert normalized_evidence in docs, (
                f'{row_id}: evidence not found in docs: '
                f'{evidence}'
            )

        if row["expected"] is not None:
            normalized_expected = normalize(row["expected"])

            for value in row["must_contain"]:
                normalized_value = normalize(value)

                assert normalized_value in normalized_expected, (
                    f'{row_id}: must_contain value "{value}" '
                    f'is not present in expected answer: '
                    f'{row["expected"]}'
                )


def main():
    docs = load_docs()
    rows = load_dataset()

    validate(rows, docs)

    print(f"{len(rows)} cases OK")


if __name__ == "__main__":
    main()
