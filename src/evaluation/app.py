import glob
import hashlib
import json
import os
import re
import yaml
import ollama
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "v1.yaml")


def load_config(path=CONFIG_PATH, overrides=None):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if overrides:
        config.update(overrides)

    return config


def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def chunk_sentences(text, max_words=150, overlap_sentences=1):
    chunks, current, count = [], [], 0
    for s in split_sentences(text):
        n = len(s.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            count = sum(len(x.split()) for x in current)
        current.append(s)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(text, source, max_words=150, overlap_sentences=1):
    sections = [s.strip() for s in re.split(r"\n\s*\n", text.strip()) if s.strip()]
    title = os.path.splitext(os.path.basename(source))[0]
    if len(sections) > 1 and "\n" not in sections[0]:
        title, sections = sections[0], sections[1:]

    out = []
    for sec in sections:
        first, _, rest = sec.partition("\n")
        if first.endswith(":") and rest.strip():
            header, body = first, rest
        else:
            header, body = "", sec
        for c in chunk_sentences(body, max_words, overlap_sentences):
            label = f"{title} | {header}".rstrip(" |") if header else title
            out.append({"source": os.path.basename(source), "text": f"{label}\n{c}"})
    return out


def embed(texts, model, batch_size=32):
    vecs = []
    for i in range(0, len(texts), batch_size):
        vecs.extend(
            ollama.embed(model=model, input=texts[i : i + batch_size])["embeddings"]
        )
    v = np.array(vecs, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


class RagApp:
    def __init__(self, config: dict):
        self.cfg = config
        self.chunks = self._load_chunks()
        self.vecs = self._embed_chunks()

    def _load_chunks(self):
        docs_dir = self.cfg["docs_dir"]
        if not os.path.isabs(docs_dir):
            docs_dir = os.path.join(BASE_DIR, docs_dir)

        paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))

        if not paths:
            raise FileNotFoundError(f"No .txt files found in '{docs_dir}/'")

        chunks = []

        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                chunks.extend(
                    chunk_document(
                        f.read(),
                        path,
                        self.cfg["chunk_size"],
                        self.cfg["overlap_sentences"],
                    )
                )

        return chunks

    def _embed_chunks(self):
        texts = [self.cfg["doc_prefix"] + chunk["text"] for chunk in self.chunks]

        key_data = [
            self.cfg["embed_model"],
            self.cfg["doc_prefix"],
            self.cfg["chunk_size"],
            self.cfg["overlap_sentences"],
            texts,
        ]

        key = hashlib.sha256(
            json.dumps(
                key_data,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:16]

        cache_dir = self.cfg["cache_dir"]

        if not os.path.isabs(cache_dir):
            cache_dir = os.path.join(BASE_DIR, cache_dir)

        os.makedirs(cache_dir, exist_ok=True)

        cache_path = os.path.join(
            cache_dir,
            f"{key}.npy",
        )

        if os.path.exists(cache_path):
            return np.load(cache_path)

        vectors = embed(
            texts,
            self.cfg["embed_model"],
        )

        np.save(cache_path, vectors)

        return vectors

    def retrieve(self, question, k=None):
        k = min(
            k or self.cfg["top_k"],
            len(self.chunks),
        )

        query = embed(
            [self.cfg["query_prefix"] + question],
            self.cfg["embed_model"],
        )[0]

        scores = self.vecs @ query

        hits = []

        for index in np.argsort(-scores)[:k]:
            if scores[index] >= self.cfg["min_score"]:
                hits.append(
                    {
                        **self.chunks[index],
                        "score": float(scores[index]),
                    }
                )

        return hits

    def answer(self, question):
        hits = self.retrieve(question)

        if not hits:
            return {
                "answer": "I don't know based on the handbook.",
                "contexts": [],
                "sources": [],
                "hits": [],
            }

        context = "\n\n".join(
            f"[{number}] (source: {hit['source']})\n" f"{hit['text']}"
            for number, hit in enumerate(hits, 1)
        )

        response = ollama.chat(
            model=self.cfg["model"],
            messages=[
                {
                    "role": "system",
                    "content": self.cfg["system_prompt"],
                },
                {
                    "role": "user",
                    "content": (f"Context:\n{context}\n\n" f"Question: {question}"),
                },
            ],
            options={
                "temperature": self.cfg["temperature"],
            },
        )

        return {
            "answer": response["message"]["content"],
            "contexts": [h["text"] for h in hits],
            "sources": sorted({h["source"] for h in hits}),
            "hits": hits,
        }


if __name__ == "__main__":
    config = load_config()
    app = RagApp(config)

    print(f"Loaded {len(app.chunks)} chunks. Ask a question (empty line to quit).")

    while True:
        q = input("\n> ").strip()

        if not q:
            break

        result = app.answer(q)

        print(result["answer"])
        print("sources:", ", ".join(result["sources"]))
