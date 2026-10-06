# LLM Evaluation Pipeline

A small, hands-on course about evaluating language models. We build a RAG
(Retrieval-Augmented Generation) app first, then learn how to evaluate it
properly, using only local models via Ollama.

Everything runs on your machine. No API keys, no cloud, no costs.

## How this course works

We start from zero: installing Ollama, pulling the models, setting up the
project. Then we build the RAG handbook assistant step by step, understand
every part of it, and after that we get to the actual topic: evaluating the
model's answers with test cases, a judge model, and metrics.

## Table of materials

1. [Introduction and setup](docs/01-introduction.md)
2. [RAG app walkthrough](docs/02-rag-app-walkthrough.md)
3. [Defining evaluation criteria](docs/03-defining-evaluation-criteria.md)
4. [Build dataset](docs/04-build-your-test-dataset.md)
5. [Code based evaluation](docs/05-code-based-scorers.md)
5. [LLM judge evaluation](docs/06-llm-as-judge-evaluation.md)

## Requirements

- Python
- Ollama with three models pulled: `gemma4:e4b` (the app under test), `qwen3.5:9b` (the judge), and `nomic-embed-text-v2-moe` (embeddings)
