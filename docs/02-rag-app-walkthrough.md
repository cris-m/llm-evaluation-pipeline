# RAG App Walkthrough

The idea is pretty simple: load the documents, split them into chunks, turn those chunks into vectors, and use the highest-scoring chunks to answer questions with a local LLM.

The important part is that **each step can fail in a different way**. Retrieval can return the wrong chunks. Chunking can split useful information apart. The LLM can ignore the context or make up an answer. Module 5 is about learning how to tell these failures apart.

The code is in [app.py](../src/evaluation/app.py). Read it top to bottom before you run it. The main flow is:

1. `RagApp._load_chunks()` reads the handbook files and calls `chunk_document()` to split them into chunks.
2. `RagApp.retrieve()` finds the chunks that are most relevant to the question.
3. `RagApp.answer()` sends those chunks to the local LLM and returns the answer along with the chunks it used.

---

## 1. The config file (`configs/v1.yaml`)

Before getting into the code, let's look at the config file:

```yaml
name: v1-baseline
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
  Use ONLY the numbered context passages provided.
  If the answer is not in them, reply exactly: "I don't know based on the handbook."
  If a specific rule overrides a general one, follow the specific rule.
  Be concise and cite the sources you used like [1], [2].
```

Here is what the settings are for:

- `docs_dir`: folder containing the handbook `.txt` files. It's a relative path, resolved against the script's own folder.
- `embed_model`: the model used to turn text into vectors.
- `model`: the LLM used to generate the final answer.
- `chunk_size`: target size of each chunk, in words.
- `overlap_sentences`: repeats the last sentence in the next chunk. This helps when an idea crosses a chunk boundary.
- `top_k`: number of chunks to retrieve for each question.
- `min_score`: minimum similarity score for a result to be included. At `0.0`, this filter is basically off.
- `temperature`: set to `0` so the answers are deterministic, which is useful when evaluating the app.
- `doc_prefix` / `query_prefix`: prefixes expected by the Nomic embedding model. Documents use `search_document: ` and questions use `search_query: `.
- `cache_dir`: where the embedding cache is stored. This is also resolved relative to the script.
- `system_prompt`: instructions given to the LLM. It tells the model to only use the provided context, say "I don't know based on the handbook." when the answer isn't there, and cite the context using `[1]`, `[2]`, etc.

---

## 2. Loading the config from YAML

At the top of the file we define where the config lives:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "v1.yaml")
```

`BASE_DIR` is the folder where the script is located. Building paths from it means the app doesn't depend on the folder you're currently in when you run it.

Then the config loader:

```python
def load_config(path=CONFIG_PATH, overrides=None):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if overrides:
        config.update(overrides)
    return config
```

`yaml.safe_load` reads the YAML and turns it into a Python dictionary.

The `or {}` is just a small guard for an empty YAML file. In that case `safe_load` returns `None`, so we use an empty dictionary instead.

---

## 3. `split_sentences(text)`

```python
return re.split(r"(?<=[.!?])\s+", text.strip())
```

This splits the text into sentences.

The `(?<=[.!?])` part means that it looks for whitespace immediately after `.`, `!`, or `?`. The punctuation stays attached to the sentence.

It's a simple sentence splitter rather than a full NLP sentence tokenizer, which is enough for these documents.

---

## 4. `chunk_sentences(text, max_words=150, overlap_sentences=1)`

```python
chunks, current, count = [], [], 0
for s in split_sentences(text):
    n = len(s.split())
    if current and count + n > max_words:
        chunks.append(" ".join(current))
        current = current[-overlap_sentences:]
        count = sum(len(x.split()) for x in current)
    current.append(s)
    count += n
```

The function goes through the document sentence by sentence and keeps track of how many words are in the current chunk.

When adding another sentence would take the chunk over the word limit, the current chunk is saved and a new one starts.

The overlap comes from:

```python
current = current[-overlap_sentences:]
```

With `overlap_sentences: 1`, the last sentence from the previous chunk is carried into the next one.

This is useful because sometimes an idea starts near the end of one chunk and continues into the next. Having the last sentence repeated gives retrieval a better chance of finding the relevant context.

The final chunk is added after the loop so any remaining text isn't lost.

---

## 5. `chunk_document(text, source, ...)`

This function adds some structure on top of the basic chunking.

First, the document is split into sections:

```python
sections = re.split(r"\n\s*\n", text.strip())
title = os.path.splitext(os.path.basename(source))[0]
if len(sections) > 1 and "\n" not in sections[0]:
    title, sections = sections[0], sections[1:]
```

The file is split on blank lines.

If the first section is just a single line, it is treated as the document title. Otherwise, the filename is used as the title.

For each section, the first line is checked:

```python
first, _, rest = sec.partition("\n")
if first.endswith(":") and rest.strip():
    header, body = first, rest
else:
    header, body = "", sec
```

If the first line ends with `:`, we treat it as a section header, for example:

```
Vacation Policy:
```

The label gives the LLM some information about where the chunk came from, and because it is part of the embedded text, it also gives the retrieval model useful section/title information.

Then each chunk gets a label:

```python
label = f"{title} | {header}".rstrip(" |") if header else title
out.append({"source": os.path.basename(source), "text": f"{label}\n{c}"})
```

So a chunk might start with something like:

```
handbook | Vacation Policy
```

That gives the LLM some information about where the chunk came from, and it also adds a few useful keywords for the embedding model.

---

## 6. `embed(texts, model, batch_size=32)`

At the end of the embedding function:

```python
v = np.array(vecs, dtype=np.float32)
return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
```

The texts are sent to Ollama in batches instead of one at a time. The default batch size is 32.

After that, the vectors are normalized.

Why do this?

Once every vector has a length of 1, the dot product between two vectors is the same as their cosine similarity. That means retrieval can later do:

```python
self.vecs @ query
```

to compare the query against all document vectors at once.

The `1e-12` is just there to avoid a division-by-zero problem if an empty vector somehow appears.

---

## 7. `RagApp.__init__(config)`

```python
def __init__(self, config: dict):
    self.cfg = config
    self.chunks = self._load_chunks()
    self.vecs = self._embed_chunks()
```

The config is loaded before the app is created.

The constructor stores the config, loads and chunks the documents, and then creates the embeddings.

This all happens once when the app starts. After that, the app is ready to handle questions.

---

## 8\. `_load_chunks()`

```
docs_dir = self.cfg["docs_dir"]
if not os.path.isabs(docs_dir):
    docs_dir = os.path.join(BASE_DIR, docs_dir)
```

The `docs_dir` from the YAML can be a relative path such as:

```
docs/handbook
```

The code resolves that path against `BASE_DIR`, so it always points to the handbook folder inside the project rather than a folder relative to wherever the terminal happens to be.

Then the `.txt` files are found:

```python
paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
if not paths:
    raise FileNotFoundError(f"No .txt files found in '{docs_dir}/'")
```

All `.txt` files are loaded in sorted order.

If there aren't any files, the app raises a clear error instead of continuing and failing somewhere later.

Each file is read and passed to `chunk_document()`. The chunks from all the files are then combined into one list.

---

## 9. `_embed_chunks()` — the cache

Embedding all the chunks can take a while, so the app caches the results.

The cache key is built from:

```python
key_data = [
    self.cfg["embed_model"],
    self.cfg["doc_prefix"],
    self.cfg["chunk_size"],
    self.cfg["overlap_sentences"],
    texts,
]
```

The idea is that anything that can change the resulting vectors should be part of the cache key.

That includes the embedding model, document prefix, chunk size, overlap setting, and the actual text being embedded.

This is useful when changing the config. For example, if you change `chunk_size` or `overlap_sentences`, the old embeddings won't accidentally be reused.

The cache directory is handled the same way as `docs_dir`:

```python
cache_dir = self.cfg["cache_dir"]
if not os.path.isabs(cache_dir):
    cache_dir = os.path.join(BASE_DIR, cache_dir)
os.makedirs(cache_dir, exist_ok=True)
```

If the cache directory doesn't exist, it gets created.

If the cache file already exists, the embeddings are loaded from it. Otherwise, the app calls `embed()`, saves the result, and uses it.

---

## 10. `retrieve(question, k=None)`

This is the retrieval part of the RAG pipeline.

First, the question is embedded:

```python
query = embed(
    [self.cfg["query_prefix"] + question],
    self.cfg["embed_model"],
)[0]
scores = self.vecs @ query
```

The `search_query: ` prefix is added before embedding the question.

Then the query vector is compared against all the document vectors with one matrix multiplication.

The results are sorted:

```python
for index in np.argsort(-scores)[:k]:
```

`np.argsort(-scores)` puts the highest scores first, and `[:k]` keeps the top `k` results.

The `min_score` setting is also used to remove results that don't meet the minimum similarity threshold. With the current value of `0.0`, that filter is effectively disabled.

Each result keeps the original chunk information along with its similarity score.

---

## 11. `answer(question)`

The first thing `answer()` does is retrieve the relevant chunks:

```python
hits = self.retrieve(question)
if not hits:
    return {"answer": "I don't know based on the handbook.", ...}
```

If nothing is retrieved, the app doesn't call the LLM. It just returns:

> I don't know based on the handbook.

This gives us a simple fallback when retrieval returns no chunks above the configured similarity threshold.

When there are results, they are numbered:

```python
context = "\n\n".join(
    f"[{number}] (source: {hit['source']})\n"
    f"{hit['text']}"
    for number, hit in enumerate(hits, 1)
)
```

The LLM sees the chunks as:

```bash
[1] (source: handbook.txt)
...

[2] (source: handbook.txt)
...
```

Those numbers match the instructions in the system prompt, so the model can cite the passages it used.

Then `ollama.chat()` gets two messages:

1. The system prompt with the rules.
2. The user message containing the retrieved context and the question.

The temperature is set to `0`, which keeps the output deterministic.

The returned result includes the answer, the retrieved contexts, the source files, and the retrieval hits, including their similarity scores. That makes it easier to inspect what happened during retrieval.

---

## 12. The `__main__` block

The app starts with:

```python
config = load_config()
app = RagApp(config)
```

The config is loaded from `configs/v1.yaml`, and then the RAG app is created.

The rest is a simple question loop:

```python
while True:
    q = input("\n> ").strip()
    if not q:
        break
```

Type a question, get an answer. Press Enter on an empty line to exit.

The older version used a one-liner with the walrus operator. This version uses a regular `while True` loop and an explicit `break`, which is a little easier to follow.

---

## The whole flow

The app loads the YAML config, reads the handbook documents, splits them into labeled chunks, embeds them, and caches the embeddings.

When a question comes in, the question is embedded and compared against all the document chunks. The top results are passed to the LLM along with instructions to only use that context and cite the passages it used.

If nothing useful is retrieved, the app returns:

> I don't know based on the handbook.

The nice part is that most of the retrieval settings live in `configs/v1.yaml`. If you want to change the chunk size or `min_score`, you can change the config and restart the app. The cache key will take care of generating new embeddings when the relevant settings change.

> **Test questions:** See [test_questions.md](src/evaluation/docs/test_questions.md) for example questions used to manually test the RAG app.
