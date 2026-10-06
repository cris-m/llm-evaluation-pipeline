import glob
import json
import os

import ollama

from evaluation.app import chunk_document, load_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "datasets", "dataset_draft.jsonl")

PROMPT = """From the passage, write 2 question/answer pairs.

For each pair, give:

- "question": a question that can be answered from the passage
- "expected": a readable reference answer
- "must_contain": a list with a short string the answer should contain for a simple code check
- "evidence": an exact quote of 3-10 words from the passage that supports the answer

For "must_contain":
- Use a short, robust string such as a number with its unit or a person's/company name.
- Do not use a whole sentence.
- For yes/no questions, return an empty list [].
- If there is no reliable short string to check, return [].

Return JSON:
{{
  "pairs": [
    {{
      "question": "...",
      "expected": "...",
      "must_contain": ["..."],
      "evidence": "..."
    }}
  ]
}}

Passage:
{p}
"""


def main() -> None:
    config = load_config()

    model = config.get("judge_model")
    if not model:
        raise SystemExit(
            "Add 'judge_model' to the config (a different model from 'model')."
        )
    if model == config["model"]:
        print(f"Warning: the question writer and the app both use {model}.")

    docs_dir = config["docs_dir"]
    if not os.path.isabs(docs_dir):
        docs_dir = os.path.join(BASE_DIR, docs_dir)

    rows, skipped = [], 0

    for path in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            chunks = chunk_document(
                f.read(), path, config["chunk_size"], config["overlap_sentences"]
            )

        for chunk in chunks:
            passage = chunk["text"]
            result = ollama.chat(
                model=model,
                format="json",
                messages=[{"role": "user", "content": PROMPT.format(p=passage)}],
            )

            try:
                pairs = json.loads(result["message"]["content"]).get("pairs", [])
            except (json.JSONDecodeError, AttributeError):
                skipped += 1
                continue

            for pair in pairs:
                if not isinstance(pair, dict):
                    continue
                evidence = pair.get("evidence", "")
                if isinstance(evidence, list):
                    evidence = evidence[0] if evidence else ""
                must_contain = pair.get("must_contain", [])
                if not isinstance(must_contain, list):
                    must_contain = [must_contain]

                if (
                    not isinstance(evidence, str)
                    or not evidence
                    or evidence not in passage
                    or not pair.get("question")
                    or not pair.get("expected")
                ):
                    skipped += 1
                    continue

                rows.append(
                    {
                        "id": f"s{len(rows):03d}",
                        "question": pair["question"],
                        "expected": pair["expected"],
                        "must_contain": [str(m) for m in must_contain],
                        "evidence": [evidence],
                        "tags": ["synthetic"],
                    }
                )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")

    print(f"Wrote {len(rows)} draft cases to {OUTPUT_PATH} ({skipped} discarded)")


if __name__ == "__main__":
    main()
