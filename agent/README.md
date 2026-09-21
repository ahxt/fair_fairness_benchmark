# Local LangChain Agent

This is a basic LangChain Deep Agent for the Fair Fairness Benchmark (FFB) repository. It uses a local Qwen model through Ollama and calls a custom tool to list the repository's reproduction scripts.

## Requirements

- Python 3.12
- Ollama
- The `qwen3:4b` Ollama model

## Setup

Run these commands from the main `fair_fairness_benchmark` folder:

```bash
ollama pull qwen3:4b
python3 -m venv .venv-agent
source .venv-agent/bin/activate
python -m pip install -r agent/requirements.txt
```

## Run

Make sure Ollama is open, then run:

```bash
python agent/basic_agent.py
```