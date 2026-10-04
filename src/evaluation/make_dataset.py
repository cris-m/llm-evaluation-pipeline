import glob
import json
import os

import ollama

from evaluation.app import load_config, chunk_document


PROMPT = """From the passage, write 2 question/answer pairs.

For each pair, give:

- "question": a question that can be answered from the passage
- "expected": a readable reference answer
- "must_contain": a short string the answer should contain for a simple code check
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


config = load_config()

docs_dir = config["docs_dir"]
if not os.path.isabs(docs_dir):
    docs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        docs_dir,
    )

rows = []

for path in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
    with open(path, "r", encoding="utf-8") as f:
        chunks = chunk_document(
            f.read(),
            path,
            config["chunk_size"],
            config["overlap_sentences"],
        )

    for chunk in chunks:
        passage = chunk["text"]

        result = ollama.chat(
            model=config["model"],
            format="json",
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(p=passage),
                }
            ],
        )

        pairs = json.loads(
            result["message"]["content"]
        ).get("pairs", [])

        for pair in pairs:
            evidence = pair.get("evidence", "")
            must_contain = pair.get("must_contain", [])

            if not evidence or evidence not in passage:
                continue

            if not isinstance(must_contain, list):
                must_contain = [must_contain]

            rows.append(
                {
                    "id": f"s{len(rows):03d}",
                    "question": pair["question"],
                    "expected": pair["expected"],
                    "must_contain": must_contain,
                    "evidence": [evidence],
                    "tags": ["synthetic"],
                }
            )


output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "datasets",
    "dataset_draft.jsonl",
)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(
        "\n".join(json.dumps(row) for row in rows)
    )

print(f"Wrote {len(rows)} draft cases to {output_path}")
