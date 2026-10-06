## Code-based evaluation

Code-based evaluation means using ordinary Python code to check the output from your RAG app.

For each test question, you run the app, get its answer and retrieved context, and then run a few small checks against that result. For example:

- Does the answer contain the number we expected?
- Did retrieval find the passage containing the evidence?
- If the question is unanswerable from the handbook, did the app say it doesn't know?

These checks are useful because they're fast, free, and deterministic. If you run the same evaluation against the same results ten times, you'll get the same scores.

The downside is that code is not very good at judging things like whether an answer is clear, whether it is well written, or whether it correctly says the same thing as a reference answer using different words. That's where the LLM judge comes in later.

### Code-based scorers

Each scorer checks one thing about one evaluation result. The scorer implementations
are in [`code_scorers.py`](../src/evaluation/code_scorers.py). They all have the same basic
shape: the function gets the test case and the app output, and returns a dictionary
containing a score between 0 and 1.

For example:

```python
def must_contain(example, output):
    required = example.get("must_contain") or []
    if not required:
        return {"score": None}

    answer = _answer(output)
    missing = [item for item in required if not _has_token(item, answer)]
    return {
        "score": 1 - len(missing) / len(required),
        "missing": missing,
    }
```

The `None` here is intentional.

Not every question has a list of key facts that can be checked with `must_contain`. In that case, the scorer skips the question instead of giving it a score of 0. Otherwise, the overall average would be lower simply because the scorer didn't apply to that question.

There are six scorers:

| Scorer               | What it checks                                                            | What it uses                      |
| -------------------- | ------------------------------------------------------------------------- | --------------------------------- |
| `must_contain`       | Does the answer contain the expected facts?                               | `must_contain` in the dataset     |
| `abstains_correctly` | Does the app refuse to answer when the docs don't contain the answer?     | the `unanswerable` tag            |
| `retrieval_hit`      | Did retrieval return at least one chunk containing the expected evidence? | `evidence`                        |
| `evidence_recall`    | What fraction of the expected evidence was retrieved?                     | `evidence`                        |
| `reciprocal_rank`    | How high was the first relevant chunk?                                    | `evidence`                        |
| `latency_ok`         | Did the request finish within the allowed time?                           | measured by the evaluation runner |

There are a couple of details about the matching that are easy to miss.

**Matching is done by whole token, not by a simple substring search.** The scorer lowercases the text, normalizes curly quotes, and removes extra whitespace before comparing it. That means `"5"` matches a standalone `5`, but not the `5` inside `"15"`. Similarly, `"HR"` should not match the `hr` inside `"three"`.

This avoids a lot of false positives that you'd get from simply doing:

```python
if required_text in answer:
```

`retrieval_hit` and `evidence_recall` are also slightly different.

Suppose a question requires two pieces of evidence, one from each of two sections. If retrieval finds one of them, `retrieval_hit` is still 1.0 because it found _a_ relevant chunk. `evidence_recall`, however, is 0.5 because only half of the expected evidence was retrieved.

For questions that depend on multiple chunks, `evidence_recall` is therefore the more useful number.

The retrieval scores are also useful when debugging answer quality. If retrieval is good but `must_contain` is low, the right information reached the model and the model still produced a bad answer. If both are low, the retriever probably missed the information in the first place.

All six scorers are registered in one dictionary at the bottom of the file:

```python
SCORERS = {
    "must_contain": must_contain,
    "abstains_correctly": abstains_correctly,
    "retrieval_hit": retrieval_hit,
    "evidence_recall": evidence_recall,
    "reciprocal_rank": reciprocal_rank,
    "latency_ok": latency_ok,
}
```

Adding another code-based check is therefore straightforward: write the scorer function and add it to this dictionary.

### Evaluation runner

The scorers don't run the RAG application themselves. [`code_evaluation.py`](../src/evaluation/code_evaluation.py)
is responsible for running the app, collecting the results, and passing them to the
scorers.

For each test case, the runner:

1. Loads the dataset.
2. Starts the RAG app using the selected config.
3. Calls `app.answer()` for each question and measures how long it takes.
4. Runs all of the scorers against the result.
5. Prints the overall scores, scores by tag, latency, and cases that need attention.
6. Saves the detailed results.

The dataset loader also does a small amount of validation. If a JSONL line is malformed, the run stops and reports the file and line number instead of failing with an unhelpful JSON error somewhere later.

There are two failure cases worth handling separately.

If the RAG app crashes on one question, that question is recorded as an error and the evaluation continues. You don't want case 37 to throw away the results from the other 69 cases.

Likewise, if one scorer has a bug, the other scorers should still run. The broken scorer is recorded as an error for that case rather than hiding all of the other scores.

For a local model, a full evaluation can take a while, so these two behaviors make a big difference when you're iterating on the system.

Run the evaluation from the project root:

```bash
uv run code_evaluation.py --output runs/v1.json --csv
```

When you're working on the evaluation code itself, it's a good idea to start with a small run:

```bash
uv run code_evaluation.py --limit 5
```

That way, a mistake in the runner costs seconds instead of waiting for the entire dataset.

You can also evaluate a particular slice of the dataset:

```bash
uv run code_evaluation.py --tag unanswerable
```

And `--config` lets you compare different RAG configurations.

### Reading a real run

Here's an example from a full 70-case run:

| Scorer               | Mean | Cases |
| -------------------- | ---- | ----- |
| `must_contain`       | 1.00 | 35    |
| `retrieval_hit`      | 1.00 | 51    |
| `evidence_recall`    | 0.99 | 51    |
| `reciprocal_rank`    | 0.99 | 51    |
| `abstains_correctly` | 1.00 | 19    |
| `latency_ok`         | 0.20 | 70    |

At first glance, this looks pretty good. Retrieval is finding the evidence, the answerable questions contain the expected facts, and the app is abstaining correctly.

The obvious problem is latency.

Only 20% of the requests met the latency target. The average response took about 23 seconds, and the slowest one took 125 seconds.

The "Cases to look at" list was therefore mostly `latency_ok` failures. That's useful, but the per-tag table is also worth checking because it can show whether the slow cases are concentrated in a particular type of question.

There is another important caveat with these numbers: don't interpret every `1.00` as meaning that part of the system is perfect.

For example, `must_contain` only applies to 35 of the 51 answerable questions in this dataset. The remaining questions don't have a useful `must_contain` check, so they are excluded from that average.

The code-based evaluation also doesn't tell us whether an answer is well written, whether it includes unnecessary information, or whether it is semantically equivalent to a reference answer while using completely different wording. It certainly doesn't catch every hallucination.

That's the boundary between these two evaluation methods. **Code-based scorers are good at checking things that can be stated as explicit rules. The LLM judge is useful for the things that are harder to express as rules.**
