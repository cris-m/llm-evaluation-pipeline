## Build your test dataset

**Why:** Your dataset defines what "good" looks like. Weak test cases can make a mediocre system look great.

**The idea:** Mix different kinds of questions:

- **Straightforward:** the answer is clear and sits in one chunk.
- **Multi-chunk:** the answer needs information from two or more chunks.
- **Unanswerable:** the docs don't say, so the right answer is "I don't know."
- **Reworded:** the same question, asked differently.
- **Tricky:** ones you expect to break the system, like yes/no traps, calculations, rules that conflict, or an instruction hidden in the question.

Tag every case. Then you can ask "how well do we handle unanswerable questions?" instead of relying on one overall score.

If an LLM writes some of the questions, review every one. It can write a question that sounds fine but has a wrong answer, and then you're grading your system against bad data.

Set aside about 20% as a holdout set and keep it out of your tuning loop. Don't look at it while you adjust prompts or retrieval settings.

**Build:** Save your test cases in `dataset.jsonl`, one JSON object per line. Each case has six fields:

| Field          | What it's for                                                                                                                                                     |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`           | A unique ID                                                                                                                                                       |
| `question`     | What you ask the system                                                                                                                                           |
| `expected`     | A readable reference answer. The judge uses it.                                                                                                                   |
| `must_contain` | Short strings the answer must include, for the code check. Leave it `[]` when a code check would be unreliable, such as yes/no answers, and let the judge decide. |
| `evidence`     | Short exact quotes from the docs that contain the answer. The retrieval check uses them.                                                                          |
| `tags`         | Labels that describe the case                                                                                                                                     |

For an unanswerable case, set `expected` to `null` and leave `must_contain` and `evidence` empty. The right behavior is to say "I don't know," not make up an answer.

Two rules keep the code checks honest:

- **Keep `must_contain` short.** Use a number with its unit, or a name. Never use `"no"`, because it matches "know", "not" and "note". And don't use a whole sentence, because it fails whenever the model rewords it.
- **Copy `evidence` from the docs.** Use a quote of 3-10 words, not the answer text. That's what lets retrieval be checked even for yes/no questions.

One example of each kind:

```json
{"id": "q001", "question": "How many vacation days do new employees get?", "expected": "15 vacation days per year", "must_contain": ["15"], "evidence": ["15 vacation days per year"], "tags": ["easy"]}
{"id": "q008", "question": "If I need a high-performance laptop for 3D CAD and want to work remotely from outside the US for 15 days, whose approvals do I need?", "expected": "Your director (for the laptop) and HR (written approval for working abroad more than 10 days)", "must_contain": ["director", "HR"], "evidence": ["requires approval from their director", "requires written approval from HR"], "tags": ["multi-chunk"]}
{"id": "q024", "question": "Can a visitor enter an active robot cell if they are constantly escorted by an employee?", "expected": "No. Visitors may never enter an active robot cell, even when escorted", "must_contain": [], "evidence": ["may not enter an active robot cell under any circumstances"], "tags": ["tricky", "yes-no"]}
```

---

## Start with a draft

You don't have to write every question yourself. Use [make_dataset.py](../src/evaluation/make_dataset.py) to get a starting point. It generates two questions from each document chunk.

After running the script, you'll find the draft here:

```bash
datasets/dataset_draft.jsonl
```

The script checks that each evidence quote actually appears in the chunk it came from.

That's just the first pass. Go through the draft yourself. Remove questions that aren't useful, fix wrong answers, get rid of duplicates, and fill in `must_contain` where it makes sense.

Then move the good cases into [dataset.jsonl](../src/evaluation/datasets/dataset.jsonl).

---

## Check the dataset

Before you use the dataset, run the check. Python script is located [check_dataset.py](../src/evaluation/check_dataset.py)

```bash
uv run python check_dataset.py
```

It catches duplicate IDs, duplicate questions, evidence that isn't in the handbook, and unanswerable cases that accidentally have evidence or `must_contain` values.

Run it again whenever you edit `dataset.jsonl`.

---

## Common pitfalls

- **All-easy datasets.** If everything scores 95%, you're not learning anything.
- **Near-duplicates.** Ten slightly different versions of one question don't give you ten tests.
- **Wrong expected answers.** This is the worst one. A correct system looks like it's failing.
- **Unreviewed synthetic cases.** They're fine for a first draft, but they shouldn't define what "correct" means.
