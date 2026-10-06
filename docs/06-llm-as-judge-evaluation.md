# LLM Judge Evaluation

Code scorers can tell you whether "15" appears in an answer. They can't tell you whether "Fifteen days of leave" means the same thing as the reference, or whether the model slipped in a claim the handbook never made. A second LLM, the judge, handles those.

The judge can be wrong too, so we check it against human scores before trusting it.

There are two judge scores:

- `correctness`
- `faithfulness`

## Setup

The judge is configured in [llm_judge_scorers.py](../src/evaluation/llm_judge_scorers.py):

```python
MODEL = "gemma4:e4b"
NUM_CTX = 16384
VOTES = 3
```

Use a different model from the one under test. A model grading its own output tends to be generous. The runner prints a warning if the two are the same.

The judge returns JSON with a short `reasoning` and a score. The reasoning doesn't change the number, but you'll want it when you're working out why a case got the score it did.

## Correctness

Correctness compares the generated answer with the reference answer. The judge sees the question, the reference and the answer, and gives a score from 1 to 5:

| Score | Meaning                                              |
| ----- | ---------------------------------------------------- |
| 5     | All key facts present and correct                    |
| 4     | Essentially correct, minor omission or imprecision   |
| 3     | Partly correct, a meaningful fact is missing         |
| 2     | Mostly wrong, with some relevant correct information |
| 1     | Wrong, unrelated, or contradicts the reference       |

Contradicting a key fact caps the score at 2. Extra correct detail isn't penalized. Scores are mapped to 0–1 (1 becomes 0.00, 5 becomes 1.00), so they sit next to the code scorers in the same table.

Wording doesn't matter. "$450" and "four hundred and fifty dollars" should get the same score.

Each case is judged three times and we keep the median. The votes use a little randomness, so a single call can wobble. The individual votes are saved with the result.

Questions without an `expected` answer are skipped and reported as unscored.

## Faithfulness

Faithfulness asks whether everything in the answer is supported by the retrieved chunks. The reference answer isn't used at all.

Correctness can miss this. Say the context says employees get 10 paid sick days, and the model answers "10 sick days, and unused days are paid out at year end." The first half is supported. The second half appears nowhere in the context, so it counts as unsupported even if it sounds plausible.

The judge works in two steps. It extracts the factual claims from the answer, then checks each claim against the context on its own. If two of three claims are supported, the score is 0.67. A claim that is true in general but not stated in the context counts as unsupported, and so does one that differs on a number or a name. Unsupported claims are saved with the result.

An answer with no factual claims, such as "I don't know", scores 1.00. Whether the refusal was the right call is a correctness question.

## When the judge breaks

Small local models sometimes return broken JSON. The scorer asks again, up to three attempts, and if it still can't get valid output the case is left unscored. We don't guess a score, because a judge failure shouldn't quietly turn into a good or bad result for your app.

The same goes for a prompt that fills the context window. If the input was probably truncated, the case is unscored.

The question, answer and retrieved chunks are passed in as tagged data, so text inside a handbook chunk can't give the judge instructions.

## Running it

```bash
uv run llm_judge_evaluation.py --output runs/v2.json --csv
```

Results are saved as JSON, plus CSV if you pass `--csv`. Without `--output`, the default is `results/llm_judge_eval_<run>_<timestamp>.json`.

Options you'll use most:

- `--limit 5`: run only the first few cases.
- `--tag tricky`: run one slice of the dataset.
- `--judge-model qwen3.5:9b`: use a different judge model.
- `--votes 1`: faster, with a noisier judge.
- `--no-save`: print the report without writing any files.

## Reading a real run

This is the summary from the 70-case run:

```
correctness       0.98  (n=51, unscored=19)
faithfulness      0.96  (n=70, unscored=0)
```

Nineteen cases are unanswerable and have no reference answer, so correctness skips them. There were no errors and no judge failures.

In the per-tag table most tags sit at or near 1.00. Two stand out:

- `reasoning`: faithfulness 0.28, but only three cases.
- `multi-chunk`: 0.90 on both scores, five cases.

The run then lists every case that needs a look. Correctness is flagged below 0.75 and faithfulness below 1.00:

- `q006`: faithfulness 0.00. Correctness is 1.00, so the answer was right, but the one claim in it, "coverage starts on April 1", was marked unsupported.
- `q037`: faithfulness 0.80. Four of five claims supported. The unsupported one was the conclusion that IT approval isn't required for a $400 course.
- `q039`: correctness 0.50. The answer said pasting confidential firmware code into an external tool is prohibited, but left out the requirement to report a mistake to security.
- `q054`: faithfulness 0.50. The claim "8% of the $100,000 base salary" was marked unsupported.
- `q055`: faithfulness 0.33. Two claims were flagged: "the hire date was June 10" and "the deadline is July 10".
- `q058`: correctness 0.50 and faithfulness 0.50. The answer gave the $500 cap but left out the manager approval. It also stated "the course costs $450", which was flagged.

Several of these flags share a cause. The `reasoning` tag has only three cases (q006, q054 and q055), so its 0.28 is a hint at most. Two patterns show up:

- Claims copied from the question. "The hire date was June 10" (q055) and "the course costs $450" (q058) come from the user's question, so a strict judge marks them unsupported.
- Computed answers. $8,000 and July 10 never appear in the handbook word for word, so a literal check on them will often fail.

If you don't want either kind penalized, say so in the faithfulness prompt, then rerun and see how the scores move.

q006 is different. The handbook contains the March 18 → April 1 example word for word, so either retrieval missed that chunk or the judge got it wrong. Open the trace to find out which.

## Calibrating the judge

Judges have habits. Yours might go easy on partly correct answers, favor long ones, or miss a requirement entirely. To find out, compare it with a human.

Start with the saved run (`runs/v2.json`) and score it yourself with `label.py`:

```bash
uv run label.py runs/v2.json
```

It walks through the answerable cases one at a time and shows the question, the reference answer and the model's answer. It doesn't show the judge's score, so you can't be nudged by it. Type a score from 1 to 5 on the same scale as the judge, `s` to skip a case, or `q` to quit.

Each label is written to `datasets/calibration.jsonl` as soon as you enter it, one JSON object per line:

```json
{
  "id": "q039",
  "question": "...",
  "expected": "...",
  "answer": "...",
  "human_score": 3
}
```

You can quit and come back later. Cases that already have a label are skipped, so the script picks up where you stopped. Cases you skipped with `s` come back next time. Cases with no reference answer never show up. To change a label, delete its line from `calibration.jsonl` and run `label.py` again. Aim for at least 30 labels, and label everything if the dataset is small.

Then compare your scores with the judge's:

```bash
uv run llm_judge_evaluation.py --calibrate datasets/calibration.jsonl
```

This doesn't read `runs/v2.json`. Each line in `calibration.jsonl` already carries the question, the reference and the answer, so the script runs the judge on those again. That takes a few judge calls per label, and the scores can differ slightly from the original run. `--judge-model` and `--votes` apply here too.

You get five numbers:

- **Exact agreement:** how often you and the judge gave the same score.
- **Within one point:** how often you were at most one apart. A 4 against a 5 is minor. A 1 against a 5 is not.
- **Spearman:** whether you both rank the answers in roughly the same order.
- **Weighted kappa:** agreement beyond chance, where close misses cost less than big ones. As a rough guideline, above 0.6 is usable, and below 0.4 suggests the rubric or judging setup needs work.
- **Mean bias:** positive means the judge is more generous than you, negative means stricter.

If all your labels are the same score, kappa and Spearman can't be computed and come out as `nan`, so label a mix of good and bad answers.

Under the numbers is a list of every case where you and the judge are two or more points apart. Start there.

## Reading the calibration results

Here is what our run printed:

```
Judge calibration: 70 cases, judge gemma4:e4b
exact agreement   0.91
within 1 point    0.99
spearman          0.92
weighted kappa    0.88
mean bias         +0.13

Differ by 2+ points:
  q030: human=1 judge=5
```

The judge gave exactly the same score as the human on 91% of cases and was almost never more than a point off. Kappa is well over 0.6, and a bias of +0.13 means it's only slightly generous.

One case doesn't fit: `q030`. The human score was 1 and the judge gave a 5. Open it before you change anything. Your label in `datasets/calibration.jsonl` has the question, the reference, the model's answer and your score. The judge's score and reasoning are in `runs/v2.json`. q030 isn't in the run's "Cases to look at" list because the judge scored it 1.00.

Then work out which of these it was:

- Your label was wrong. Delete that line and run `label.py` again.
- The `expected` answer was wrong. Fix it in `dataset.jsonl`, rerun the evaluation, then delete the old line in `calibration.jsonl` and relabel. The old line keeps a copy of the wrong reference.
- The judge missed something. Add it to the rubric. For example, if the reference needs two conditions and the answer gives only one, say that every condition must be present for a 5. You don't need a new run, because `--calibrate` re-judges the labeled answers.

Rerun `--calibrate` afterward and check that the fix didn't break other cases.

## Watch out

- Don't use the same model for the judge and the app.
- Don't skip calibration.
- Don't look at the judge's scores before you label.
- Keep the expected answer out of the faithfulness prompt. Faithfulness is about the context.

## Done when

You've labeled at least 30 cases, run `--calibrate`, read every case in the "Differ by 2+ points" list, and fixed what caused each one, whether that was the rubric, a label or the dataset.
