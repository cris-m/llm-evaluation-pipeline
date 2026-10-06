# Introduction

This course is about evaluating models. Before anything else, let's get everything set up.

## What you need

- Python
- Ollama (we run everything locally, so no API keys)

## Installation

**1. Install Ollama** from [ollama.com](https://ollama.com). Gemma 4 and Qwen 3.5 need a recent build, so update if yours is old:

```bash
ollama --version
```

**2. Pull the models:**

```bash
ollama pull gemma4:e4b                 # the app we're testing
ollama pull qwen3.5:9b                 # the judge
ollama pull nomic-embed-text-v2-moe    # embeddings
```

Why two chat models? Because the judge shouldn't grade its own homework. A different family means less bias. The embedding model is just there for retrieval.

**3. Set up the project** with [uv](https://docs.astral.sh/uv):

```bash
mkdir evaluation && cd evaluation
uv init
uv venv
uv add ollama numpy pandas pydantic pyyaml pytest streamlit scikit-learn
source .venv/bin/activate              # Windows: .venv\Scripts\activate
cd src/evaluation && mkdir docs configs datasets
```

Quick rundown of the packages: `ollama` talks to the models, `numpy` and `pandas` for scores and tables, `pydantic` and `pyyaml` for config, `pytest` runs the tests, `streamlit` shows the results, `scikit-learn` for metrics.
