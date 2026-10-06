# Diagnosing Failures

A drop in score tells you something broke, not where. A RAG app finds passages, then the model answers from them. If the right passage never reached the model, no prompt change will help. If it did, a bigger `top_k` won't.

So for each failing case, ask two things in order. Did the evidence make it into the retrieved chunks? If so, what did the model do with it? Retrieval goes first because a missing chunk often makes the model invent something, which looks like low faithfulness.

## What you need

A judge run such as `runs/v2.json`, and the dataset. The run holds the judge scores and the retrieved chunks, so the script measures retrieval itself with `evidence_recall`. A `code_evaluation.py` run has no judge scores, and `calibration.jsonl` has no chunks, so neither works.

## What the script does

[diagnose.py](../src/evaluation/diagnose.py) gives each case one label. It doesn't call a model. The checks run top to bottom and the first match wins:

| Label | What happened | Where to look |
| ----- | ------------- | ------------- |
| `app error` | The app crashed on this case | The error in the run file |
| `answered unanswerable` | The docs don't have the answer and the app answered anyway | System prompt, `min_score` |
| `judge failed` | The judge produced no correctness score | Judge output, `--votes`, judge model |
| `ok` | Correctness is 0.75 or higher, or an unanswerable question was refused | Nothing |
| `retrieval miss` | Wrong answer, none of the evidence retrieved | Chunk size, `top_k`, embedding model |
| `retrieval partial` | Wrong answer, only some of the evidence retrieved | `top_k`, how multi-chunk answers are split |
| `over-abstained` | All the evidence arrived and the app still said "I don't know" | System prompt, `min_score` |
| `unfaithful` | All the evidence arrived, but the answer has unsupported claims | Stricter prompt, stronger model |
| `incomplete or misread` | All the evidence arrived, the answer is faithful, and it's still wrong or incomplete | Stronger model, a prompt that asks for every condition |

`CORRECT_AT` (0.75) and `FAITHFUL_AT` (0.5) sit at the top of the file. The faithfulness one is lower on purpose, so `unfaithful` only fires on clear drift.

Under each failure it prints the missing evidence and any unsupported claims. The last line lists cases labeled `ok` that still scored below 1.00 on faithfulness, which would otherwise vanish into the `ok` count.

## Running it

```bash
uv run diagnose.py runs/v2.json
```

Add `-v` to also see the retrieved files and the judge's reasoning. The reasoning is a paragraph per case, so it's off by default.

## Reading a real run

```bash
ok                      68
incomplete or misread   1
retrieval partial       1

Failures:
  q039: incomplete or misread  correctness=0.5  faithfulness=1.0
  q058: retrieval partial  correctness=0.5  faithfulness=0.5
      missing evidence: between $300 and $1,200
      unsupported: The AI certification course costs $450.

Correct, but faithfulness below 1.00: q006, q037, q054, q055
```

Two failures is too few for percentages.

**q058** needs the $500 AI cap and the manager approval above $300. The approval quote never arrived, so no prompt could have produced it. With `-v`, `learning_and_development.txt` shows up in the sources, so the file was retrieved and the right chunk wasn't. Run `app.retrieve()` on the question to see where it ranked. If it was sixth with `top_k` at five, raise `top_k`. If it was far down, look at chunking or the embedding model instead. The `$450` line is separate: it comes from the question, so the judge flagged something the handbook would never say.

**q039** had every quote retrieved. The answer gave the prohibition and left out the duty to report a mistake to security, which is in the same sentence of the handbook. That's the model's miss. Try asking for every requirement that applies.

The four cases on the last line are correct answers with a flagged claim. Many are claims copied from the question or computed values like $8,000. **q006:** [what you found when you opened its contexts].

## Watch out

- A wrong judge score becomes a wrong diagnosis, so calibrate the judge first.
- The refusal check is a list of phrases, so a refusal worded differently can be mislabeled.

## Done when

You can give the counts, not percentages. For this run that's one retrieval partial, one incomplete or misread, and 68 ok. You've also opened every case in both lists.