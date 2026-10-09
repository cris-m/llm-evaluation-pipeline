# Automate Evaluations and Block Regressions

A manual evaluation tells you how one run went. It says nothing when a change next week quietly makes things worse. This chapter adds three tools to catch regressions and investigate failures:

- `gate.py` compares scorer averages against minimums and exits with status 1 if any check fails.
- `.github/workflows/eval.yml` runs a small evaluation on pull requests and pushes to `main`.
- `dashboard.py` is a Streamlit app for browsing runs, comparing them, and working out why a case failed.

---

## The gate

A floor is the lowest average you'll accept for a scorer. The gate compares each required scorer against its floor and exits 0 if all floors are met and no application errors occur; otherwise it exits 1. A scorer that appears in the floors file but is missing from the run also counts as a failure.

### Setting the floors

Don't pick floors out of thin air. Take them from a run you trust. Set them too high and the gate fails for no reason; set them too low and real regressions walk straight through.

We'll use v4, the run with `top_k: 6`:

```bash
uv run gate.py runs/v4.json --write-floors --margin 0.05
```

Example output:

```bash
Wrote gate.yaml:
  must_contain          0.95   (this run: 1.000)
  retrieval_hit         0.95   (this run: 1.000)
  evidence_recall       0.95   (this run: 1.000)
  reciprocal_rank       0.93   (this run: 0.987)
  correctness           0.94   (this run: 0.990)
  faithfulness          0.91   (this run: 0.969)
  abstains_correctly    0.95   (this run: 1.000)
```

The file it writes is `configs/gate.yaml`:

```yaml
must_contain: 0.95
retrieval_hit: 0.95
evidence_recall: 0.95
reciprocal_rank: 0.93
correctness: 0.94
faithfulness: 0.91
abstains_correctly: 0.95
```

Each floor is the baseline average minus the margin, rounded down to the configured precision. Commit the file so every future run is judged by the same numbers, and edit individual floors by hand if one needs to be different.

`latency_ok` is left out on purpose. It scored 0.00 on every case in these runs, because the 15-second limit was a guess that never matched reality. A floor on a scorer that always fails tells you nothing.

### Choosing a margin

It helps to think in cases, not percentages. If a scorer averages binary results over $N$ cases, flipping one case moves the average by $1/N$.

`abstains_correctly` has 19 cases. One wrong abstention moves it by about 0.053, from 1.000 to 0.947, which drops below the 0.95 floor. That's what we want: a single invented answer to an unanswerable question deserves a look.

`correctness` has 51 cases. Between v1 and v2, q039 slipped from 0.75 to 0.50, which moved the average by roughly 0.005. A 0.05 margin absorbs wobble like that and still flags anything bigger.

Treat these as worked examples, not rules. Judge scores are noisy, so run the baseline a few times to see how much it normally moves. If harmless fluctuation keeps failing the gate, loosen the floor. If real regressions slip past it, tighten it.

### Checking a different run

Don't check a run against floors generated from that same run. The floors sit below its averages, so it will pass almost by definition.

Check v3 instead (`top_k: 4`) against the floors from v4:

```bash
uv run gate.py runs/v3.json
```

Example output:

```bash
scorer                   mean  floor
must_contain            1.000   0.95  ok
retrieval_hit           1.000   0.95  ok
evidence_recall         0.990   0.95  ok
reciprocal_rank         0.987   0.93  ok
correctness             0.985   0.94  ok
faithfulness            0.957   0.91  ok
abstains_correctly      1.000   0.95  ok

app errors: 0
Gate passed.
```

It passes, which matches the earlier comparison: on this dataset, v1 and v2 barely differ.

The gate fails on a score below its floor, on an application error, or on a missing scorer. That last rule matters. Without it, a partial run could skip the metrics you care about and still come back green.

Here's what that looks like. `runs/v2.json` from the judge chapter only has `correctness` and `faithfulness`:

```bash
scorer                   mean  floor
must_contain             none   0.95  FAIL (not scored in this run)
retrieval_hit            none   0.95  FAIL (not scored in this run)
evidence_recall          none   0.95  FAIL (not scored in this run)
reciprocal_rank          none   0.93  FAIL (not scored in this run)
correctness             0.975   0.94  ok
faithfulness            0.959   0.91  ok
abstains_correctly       none   0.95  FAIL (not scored in this run)

app errors: 0
GATE FAILED: must_contain, retrieval_hit, evidence_recall, reciprocal_rank, abstains_correctly
```

That failure is correct. The file doesn't contain every required score. For the full gate, use runs produced by `run_eval.py`.

| Option | What it does |
| --- | --- |
| `--floors PATH` | Use a different floors file (default `configs/gate.yaml`) |
| `--write-floors` | Write floors from this run instead of checking it |
| `--margin N` | Subtract this margin from baseline averages when writing floors (default `0.05`) |

### Test the gate with a broken prompt

A gate you've never seen fail isn't one you should trust yet. So break something on purpose.

Save this as `configs/bad.yaml`. It's the v1 setup, except the prompt tells the model to guess whenever the handbook is silent:

```yaml
name: bad-prompt
docs_dir: docs/handbook
embed_model: nomic-embed-text-v2-moe
model: qwen3.5:9b
judge_model: gemma4:e4b
chunk_size: 150
overlap_sentences: 1
top_k: 4
min_score: 0.0
temperature: 0
doc_prefix: "search_document: "
query_prefix: "search_query: "
cache_dir: .rag_cache
system_prompt: |
  You answer questions about the Acme Robotics Employee Handbook.
  Always give a confident, specific answer with a number or a date.
  If the context doesn't say, make a reasonable guess. Never say you don't know.
```

Run the first twenty cases and check the result:

```bash
uv run run_eval.py --config configs/bad.yaml --limit 20 --output runs/bad.json
uv run gate.py runs/bad.json
```

Example output:

```bash
scorer                   mean  floor
must_contain            1.000   0.95  ok
retrieval_hit           1.000   0.95  ok
evidence_recall         1.000   0.95  ok
reciprocal_rank         1.000   0.93  ok
correctness             1.000   0.94  ok
faithfulness            0.600   0.91  FAIL
abstains_correctly      0.000   0.95  FAIL

app errors: 0
GATE FAILED: faithfulness, abstains_correctly
```

The exit code is 1.

Those twenty cases are ten answerable questions and ten unanswerable ones. The guessing prompt still gets the answerable ones right, but it never abstains on the rest, so `abstains_correctly` drops to 0.000. The judge also gave faithfulness 0.00 to seven of the ten unanswerable cases, which backs up the picture of unsupported answers. (Faithfulness and abstention measure different things, though, so treat that as supporting evidence, not proof.)

Only `faithfulness` and `abstains_correctly` fell below their floors. Retrieval didn't change, because the prompt doesn't touch it, and `must_contain` and `correctness` stayed at 1.000. If correctness were your only gate, this regression would have shipped.

One more thing about this run: it was slow. It averaged 319 seconds per case, and the slowest case took 35 minutes, against 49 seconds per case for v3. I didn't track down why, so don't read it as a general property of guessing prompts.

If you want something faster, run three unanswerable cases with only the abstention scorer:

```bash
echo "abstains_correctly: 0.95" > /tmp/floors.yaml

uv run run_eval.py --config configs/bad.yaml --tag unanswerable --limit 3 \
  --scorers abstains_correctly --output runs/bad_quick.json

uv run gate.py runs/bad_quick.json --floors /tmp/floors.yaml
```

That's a smoke test. It doesn't replace the full run.

---

## CI

The local gate only works if someone remembers to run it. CI runs a smaller evaluation automatically on the events the workflow is configured for, catching basic regressions before a change merges.

The kinds of breakage it's good at:

- Retrieval starts returning fewer relevant chunks.
- A prompt edit stops the model from abstaining on unanswerable questions.
- A refactor makes `answer()` crash on some case.
- A handbook edit deletes text the dataset depends on.

The CI job runs the real pipeline on a clean machine, with a small model and a representative slice of the dataset.

### What CI can and cannot catch

| CI can catch | CI generally can't catch reliably |
| --- | --- |
| Retrieval regressions (`retrieval_hit`, `evidence_recall`) | Subtle drops in answer correctness |
| Failures to abstain on unanswerable questions | Unsupported claims that need the judge |
| Application errors on evaluated cases | Small quality shifts that need the judge |
| Required strings missing from answers | Regressions in cases that aren't in the subset |

A green CI check means the smoke test passed, nothing more. It doesn't tell you the answers are good. Before merging a change to the prompt, retrieval, or model, run the full evaluation with the judge.

There's one more catch. CI reads `ci.yaml`, not `v1.yaml` or `v2.yaml`. Change a setting in either of those and CI won't notice unless you make the same change in `ci.yaml`. You still need the local full run to test the configuration you plan to ship.

### Configure CI and establish its floors

The CI config uses `llama3.2:1b` and the smaller `nomic-embed-text` embedding model. A model that small behaves differently from the one in your full evaluation, so it gets its own floors, generated from its own baseline.

Make sure both models are pulled locally, then run:

```bash
uv run run_eval.py --config configs/ci.yaml --dataset datasets/dataset.jsonl --workers 1 \                                                                             
  --scorers must_contain,abstains_correctly,retrieval_hit,evidence_recall \
  --output runs/ci_baseline.json

uv run gate.py runs/ci_baseline.json --write-floors \
  --floors configs/gate_ci.yaml --margin 0.1
```

The margin is 0.1. With 20 binary-scored cases, one case changes the average by 0.05, so this margin allows roughly two case-level changes. Check it against repeated baseline runs before relying on it.

### The workflow

`eval.yml` runs on pull requests, on pushes to `main`, and on manual dispatch from the Actions tab. It does the following:

1. Installs the project with `uv sync`.
2. Runs `check_dataset.py`.
3. Installs and starts Ollama, then pulls the two small models.
4. Runs the gate.
5. Uploads `runs/ci.json`, even when the gate fails.

This lets you inspect failed cases from the artifact without rerunning the evaluation.

A red workflow only blocks a merge if GitHub is set up to require it. Turn on branch protection for `main` and mark the `eval` check as required. If you commit straight to `main` and skip pull requests, you can drop the `pull_request:` trigger.

The workflow commands run inside `src/evaluation`. If your repository is laid out differently, change `working-directory` and the artifact path. Make sure everything it needs is committed: `pyproject.toml`, `uv.lock`, the handbook, the dataset, `ci.yaml`, and both floors files.

---

## The dashboard

The terminal is fine for checking a score. For reading answers and comparing runs, a browser is easier.

Start it with:

```bash
uv run streamlit run dashboard.py
```

It reads the JSON files in `runs/`. There's no database, and nothing is rescored. It reuses the same functions as `compare.py` and `diagnose.py`:

```python
from evaluation.compare import changed_cases, compare, config_diff, run_warnings
from evaluation.diagnose import diagnose
```

Pick a runs folder and a run in the sidebar. There are three tabs:

| Tab | What it shows |
| --- | --- |
| Overview | Scorer averages, latency in seconds, and a by-tag table |
| Compare | Two runs side by side: config changes, warnings, interval results, and changed cases |
| Cases | Each question with its reference, answer, judge reasoning, unsupported claims, retrieved chunks, and diagnosis |

The Cases tab opens filtered to failures. If a run only has code-based scores, it says no diagnosis is available instead of calling every case a judge failure.

Row labels in the by-tag table include the case count, like `reasoning (3)`. Look at that number before you trust a score: a group of three swings hard on a single case. Latency is shown in seconds because `latency_ok` isn't informative in these runs.

---

## Watch out

- **Trust your baseline.** Repeat it a few times to see how much it varies, especially the judge scores.
- **Give noisy metrics room.** If the judge isn't well calibrated, faithfulness may need a wider margin.
- **Regenerate floors when the dataset changes.** New cases can make the old thresholds meaningless.
- **Investigate before relaxing a floor.** Use `diagnose.py` or the Cases tab to tell a real regression from noise.
- **Turn real failures into dataset cases.** A tagged case for a bug you actually hit makes it easy to catch next time.
- **Feed the gate the right runs.** It expects output from `run_eval.py` containing every required scorer.
- **Keep `ci.yaml` in sync.** Edits to `v1.yaml` or `v2.yaml` don't reach CI on their own.

## Done when

- `gate.py` exits 0 on a good run and 1 on the deliberately broken prompt.
- A pull request or push to `main` triggers the workflow and reports pass or fail.
- You've opened a failing case in the dashboard and read its diagnosis.