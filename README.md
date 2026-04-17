# Cirrus — LLM Agent Benchmark Framework

**Language / 语言：** English | [中文](docs/README_zh.md)

Cirrus is a modular benchmark framework for evaluating Large Language Model (LLM) agents in realistic customer-service simulation scenarios. It supports multi-turn dialogue, tool-calling tasks, multi-provider LLM backends, and automated scoring.
Our paper on arXiv: CirrusBench: Evaluating LLM-based Agents Beyond Correctness in Real-World Cloud Service Environments https://arxiv.org/abs/2603.28569 .
---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running the Benchmark](#running-the-benchmark)
- [Evaluation & Metrics](#evaluation--metrics)
- [Supported LLM Providers](#supported-llm-providers)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

Cirrus evaluates LLM agents across two task domains:

| Domain | Description |
|--------|-------------|
| `no_tool` | Pure dialogue tasks — the agent must handle user requests through conversation only |
| `with_tool` | Tool-augmented tasks — the agent may invoke external tools (API calls, database queries, etc.) |

Each task runs for multiple independent trials. Results are scored automatically and aggregated into standard benchmark metrics.

---

## Architecture

```
Orchestrator
├── Agent (LLM-backed)          ← the model under evaluation
├── Environment                 ← manages tool execution and state
└── Evaluator                   ← scores each turn and the final outcome
```

Key modules:

| Module | Path | Purpose |
|--------|------|---------|
| Orchestrator | `src/cirrus/orchestrator/` | Drives the simulation loop |
| LLM Service | `src/cirrus/llm/` | Unified multi-provider LLM interface |
| Agent | `src/cirrus/agent/` | LLM agent implementation |
| Environment | `src/cirrus/environment/` | Tool execution sandbox |
| Evaluator | `src/cirrus/evaluation/` | Per-turn and final-outcome scoring |
| Judge | `src/cirrus/judge/` | LLM-as-judge content scoring |
| Metrics | `src/cirrus/metrics/` | pass^k and reward aggregation |
| Data Models | `src/cirrus/data_model/` | Pydantic schemas for tasks, messages, tools |

---

## Quick Start

### Requirements

- Python 3.10+
- Linux / macOS / Windows

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd benchmark_public

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Set up API keys
cp .env.template .env
# Edit .env and fill in the keys you need
```

---

## Configuration

### API Keys (`.env`)

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key
QWEN_API_KEY=your_qwen_api_key          # Alibaba Cloud DashScope
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

Only the keys for providers you actually use are required.

### Model Configuration (`configs/providers_config.yaml`)

```yaml
deepseek:
  base_url: "https://api.deepseek.com"
  base_url_env: "DEEPSEEK_BASE_URL"
  api_key_env: "DEEPSEEK_API_KEY"
  default_model: "deepseek-chat"

openai:
  base_url: "https://api.openai.com/v1"
  base_url_env: "OPENAI_BASE_URL"
  api_key_env: "OPENAI_API_KEY"
  default_model: "gpt-3.5-turbo"

qwen:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  base_url_env: "QWEN_BASE_URL"
  api_key_env: "QWEN_API_KEY"
  default_model: "qwen-turbo"

claude:
  base_url: "https://api.anthropic.com"
  base_url_env: "ANTHROPIC_BASE_URL"
  api_key_env: "ANTHROPIC_API_KEY"
  default_model: "claude-3-sonnet-20240229"
```

For detailed LLM provider configuration and Judge setup, see [LLM_AND_JUDGE_GUIDE.md](docs/LLM_AND_JUDGE_GUIDE.md).

---

## Running the Benchmark

```bash
python -m cirrus.run \
  --domain no_tool \
  --model-name deepseek-chat \
  --num-tasks 10 \
  --num-trials 3 \
  --max-steps 20 \
  --max-errors 10 \
  --max-concurrency 3 \
  --save-to outputs/results.json
```

### Key CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--domain` | `no_tool` | Task domain: `no_tool` or `with_tool` |
| `--model-name` | `deepseek-chat` | LLM model to evaluate |
| `--task-ids` | — | Run specific task IDs (e.g. `--task-ids 0008 0009`) |
| `--num-tasks` | — | Run only the first N tasks |
| `--num-trials` | `3` | Independent trials per task |
| `--max-steps` | `20` | Max simulation steps per trial |
| `--max-errors` | `10` | Max consecutive errors before aborting a trial |
| `--max-concurrency` | `3` | Parallel simulations |
| `--save-to` | — | Output path for results JSON |
| `--overwrite` | `false` | Overwrite existing results |

---

## Evaluation & Metrics

After running, compute metrics with:

```bash
python -m cirrus.metrics.agent_metrics --results outputs/results.json
```

### Metrics

| Metric | Description |
|--------|-------------|
| `avg_reward` | Mean reward across all trials (1.0 = fully successful) |
| `pass^k` | Probability that at least k out of n trials succeed (from [pass^k paper](https://arxiv.org/pdf/2406.12045)) |
| `avg_agent_cost` | Mean token cost per trial |

The **judge** module uses an LLM-as-judge approach to score content quality (fluency + task completion) at each turn.

---

## Supported LLM Providers

| Provider | Example Models | Tool Calling |
|----------|---------------|:---:|
| DeepSeek | `deepseek-chat`, `deepseek-reasoner` | ✅ |
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` | ✅ |
| Qwen (DashScope) | `qwen-turbo`, `qwen-plus`, `qwen-max` | ✅ |
| Anthropic Claude | `claude-3-haiku`, `claude-3-5-sonnet` | ✅ |
| Google Gemini | `gemini-pro`, `gemini-1.5-pro` | ⚠️ Partial |

To add a new provider, subclass `BaseLLMProvider` in `src/cirrus/llm/providers/base.py` and register it in the LLM registry.

---

## Project Structure

```
benchmark_public/
├── configs/
│   ├── providers_config.yaml  # LLM provider definitions
│   ├── judge_config.yaml      # Judge model configuration
│   └── models.yaml            # Model definitions
├── data/
│   ├── prompts/               # System prompts for agent, judge, user simulator
│   └── raw_data/              # Task datasets (.jsonl)
├── outputs/                   # Simulation results
├── scripts/                   # Helper scripts
├── src/cirrus/
│   ├── agent/                 # LLM agent
│   ├── configs/               # Path and runtime config
│   ├── data_model/            # Pydantic schemas
│   ├── environment/           # Tool execution environment
│   ├── evaluation/            # Turn-level evaluators
│   ├── judge/                 # LLM-as-judge scoring
│   ├── llm/                   # Multi-provider LLM service
│   ├── metrics/               # Benchmark metric computation
│   ├── orchestrator/          # Simulation orchestrator
│   ├── utils/                 # Shared utilities
│   └── run.py                 # Entry point
├── tests/
├── .env.template
├── pyproject.toml
└── requirements.txt
```

---

## License

This project is licensed under the MIT License.

> **Note:** Keep your `.env` file out of version control. Never commit API keys.
