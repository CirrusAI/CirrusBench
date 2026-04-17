# LLM Provider & Judge Configuration Guide

**Language / 语言：** English | [中文](LLM_AND_JUDGE_GUIDE_zh.md)

This document describes how to configure and use LLM providers in this project, as well as how to configure and test the Judge scoring module.

---

## Table of Contents

- [1. LLM Providers](#1-llm-providers)
  - [1.1 Supported Providers](#11-supported-providers)
  - [1.2 Configuration File: providers_config.yaml](#12-configuration-file-providers_configyaml)
  - [1.3 Environment Variables (.env)](#13-environment-variables-env)
  - [1.4 Call Interface](#14-call-interface)
  - [1.5 Response Schema](#15-response-schema)
  - [1.6 Tool Calling](#16-tool-calling)
  - [1.7 Automatic Provider Inference](#17-automatic-provider-inference)
  - [1.8 Convenience Wrappers](#18-convenience-wrappers)
  - [1.9 Testing LLM Providers](#19-testing-llm-providers)
- [2. Judge Scoring Module](#2-judge-scoring-module)
  - [2.1 Overview](#21-overview)
  - [2.2 Configuration File: judge_config.yaml](#22-configuration-file-judge_configyaml)
  - [2.3 Judge Prompt](#23-judge-prompt)
  - [2.4 Call Interface: scoring_content](#24-call-interface-scoring_content)
  - [2.5 Return Values](#25-return-values)
  - [2.6 Testing the Judge Module](#26-testing-the-judge-module)
- [3. Adding a New Provider](#3-adding-a-new-provider)

---

## 1. LLM Providers

### 1.1 Supported Providers

| Provider | Identifier | Protocol |
|----------|------------|----------|
| DeepSeek | `deepseek` | OpenAI-compatible |
| OpenAI | `openai` | OpenAI native |
| Qwen (Alibaba DashScope) | `qwen` | OpenAI-compatible |
| Anthropic Claude | `claude` | OpenAI-compatible proxy |

All providers are accessed through the unified `call_llm` interface backed by the OpenAI SDK.

---

### 1.2 Configuration File: providers_config.yaml

Path: `configs/providers_config.yaml`

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

**Field descriptions:**

| Field | Description |
|-------|-------------|
| `base_url` | Hardcoded default API endpoint — used as the fallback value |
| `base_url_env` | Name of the environment variable whose value overrides `base_url` when set (useful for proxies or custom deployments) |
| `api_key_env` | Name of the environment variable that holds the API key |
| `default_model` | Default model name for this provider, used when running tests without an explicit model |

**`base_url` resolution priority:**

```
Explicit base_url argument  >  Environment variable (base_url_env)  >  Config file base_url
```

---

### 1.3 Environment Variables (.env)

Copy the template and fill in the API keys you need:

```bash
cp .env.template .env
```

Edit `.env`:

```env
# DeepSeek
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Qwen (Alibaba Cloud DashScope)
QWEN_API_KEY=your_qwen_api_key_here

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: override base_url (e.g. for a proxy or local deployment)
# DEEPSEEK_BASE_URL=http://your-proxy/v1
```

> `.env` is listed in `.gitignore` — do not commit it to version control.

---

### 1.4 Call Interface

Core entry point: `cirrus.llm.service.call_llm`

```python
from cirrus.llm.service import call_llm

response = call_llm(
    messages=[{"role": "user", "content": "Hello"}],
    model="deepseek-chat",
    provider="deepseek",   # optional — inferred from model name if omitted
    tools=None,            # optional — list of tool definitions
    base_url=None,         # optional — overrides the provider's default base_url
    api_key=None,          # optional — overrides the API key from the environment
    max_tokens=1024,       # any other OpenAI-compatible kwargs are forwarded as-is
    temperature=0.7,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `List[Dict]` | OpenAI-format message list |
| `model` | `str` | Model name |
| `provider` | `str` \| `None` | Provider identifier; auto-inferred when `None` |
| `tools` | `List[Dict]` \| `None` | Function-calling tool definitions |
| `base_url` | `str` \| `None` | Overrides the provider's default endpoint |
| `api_key` | `str` \| `None` | Overrides the API key from the environment |
| `**kwargs` | — | Additional arguments forwarded to the OpenAI SDK (`temperature`, `max_tokens`, etc.) |

---

### 1.5 Response Schema

`call_llm` returns a unified `LLMResponse` object:

```python
@dataclass
class LLMResponse:
    content: Optional[str]           # Text reply content
    tool_calls: List[ToolCall]       # Tool call list (empty list when none)
    usage: Optional[Usage]           # Token usage
    finish_reason: Optional[str]     # Stop reason ("stop" / "tool_calls" / "length" / etc.)
    raw_response: Any                # Raw OpenAI response object

    @property
    def has_tool_calls(self) -> bool: ...  # True if the response contains tool calls

@dataclass
class Usage:
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]

@dataclass
class ToolCall:
    id: str
    type: str
    function: ToolFunction

@dataclass
class ToolFunction:
    name: str
    arguments: str                   # Raw JSON string
    arguments_dict: Optional[Dict]   # Parsed dictionary
```

---

### 1.6 Tool Calling

```python
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a given city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
            },
            "required": ["city"]
        }
    }
}

# Turn 1: ask the model to call the tool
resp = call_llm(
    messages=[
        {"role": "system", "content": "You are a weather assistant. Call get_weather when asked about weather."},
        {"role": "user", "content": "What is the weather in Beijing today?"}
    ],
    model="deepseek-chat",
    provider="deepseek",
    tools=[WEATHER_TOOL],
)

if resp.has_tool_calls:
    tool_call = resp.tool_calls[0]
    city = tool_call.function.arguments_dict["city"]

    # Execute the tool and get the result
    weather_result = f"{city}: Sunny, 25°C"

    # Turn 2: feed the tool result back to the model
    final_resp = call_llm(
        messages=[
            ...,
            {
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": weather_result,
            }
        ],
        model="deepseek-chat",
        provider="deepseek",
    )
    print(final_resp.content)
```

---

### 1.7 Automatic Provider Inference

When `provider` is not supplied, `call_llm` infers the provider from the model name:

| Model name contains | Inferred provider |
|---------------------|-------------------|
| `deepseek` | `deepseek` |
| `gpt-`, `gpt4`, `o1-` | `openai` |
| `qwen`, `qwen2` | `qwen` |
| `claude-`, `claude` | `claude` |
| anything else | `deepseek` (default) |

```python
# provider is inferred as openai automatically
resp = call_llm(messages=[...], model="gpt-4o")
```

---

### 1.8 Convenience Wrappers

```python
from cirrus.llm.service import call_deepseek, call_openai, call_qwen, call_claude

# Equivalent to call_llm(..., provider="deepseek", model="deepseek-chat")
resp = call_deepseek(messages=[...])

# Equivalent to call_llm(..., provider="openai", model="gpt-3.5-turbo")
resp = call_openai(messages=[...])

resp = call_qwen(messages=[...])
resp = call_claude(messages=[...])
```

---

### 1.9 Testing LLM Providers

#### Integration Tests (real API calls)

File: `tests/test_providers_integration.py`

These tests read all providers from `providers_config.yaml` and make real API calls to each one. **Providers whose API key environment variable is not set are skipped automatically** — they do not cause test failures.

```bash
# Run all integration tests
pytest tests/test_providers_integration.py -v -s

# Run only a specific provider
pytest tests/test_providers_integration.py -v -s -k "deepseek"

# Run only the connectivity test
pytest tests/test_providers_integration.py -v -s -k "test_basic_connectivity"
```

**Test cases included:**

| Test name | Description |
|-----------|-------------|
| `test_show_providers_config` | Prints a config overview and checks which providers have their key set (no API call) |
| `test_basic_connectivity` | Basic dialogue connectivity test |
| `test_tool_call` | End-to-end tool-calling test including a multi-turn conversation |
| `test_usage_info` | Validates the token usage information format |
| `test_multi_turn` | Tests that context is preserved across multiple turns |

**Quick config check (no API calls):**

```bash
pytest tests/test_providers_integration.py::test_show_providers_config -v -s
```

Example output:
```
providers_config.yaml has 4 configured providers:
  [deepseek] model=deepseek-chat, key=DEEPSEEK_API_KEY (✓ set)
  [openai]   model=gpt-3.5-turbo, key=OPENAI_API_KEY  (✗ not set)
  [qwen]     model=qwen-turbo,    key=QWEN_API_KEY    (✓ set)
  [claude]   model=claude-3-sonnet-20240229, key=ANTHROPIC_API_KEY (✗ not set)
```

---

## 2. Judge Scoring Module

### 2.1 Overview

The Judge module evaluates the content coverage relationship between two responses: it determines **whether Response B substantively contains the core solution present in Response A**, and returns a quantified score.

Typical use case: in benchmark evaluation, use the reference answer as Response A and the model output under test as Response B. The LLM judge scores how well the output covers the reference.

---

### 2.2 Configuration File: judge_config.yaml

Path: `configs/judge_config.yaml`

```yaml
judge:
  model: deepseek-chat
  temperature: 0.0
```

**Field descriptions:**

| Field | Description |
|-------|-------------|
| `model` | Model used by the judge; must have a matching provider in `providers_config.yaml` |
| `temperature` | Inference temperature; set to `0.0` for stable, deterministic output |

When `scoring_content` is called without an explicit `model` or `llm_args`, these defaults are read automatically.

---

### 2.3 Judge Prompt

Path: `data/prompts/judge/judge_prompt.md`

The judge prompt defines the LLM's role and decision logic:

- **Role**: Senior Information Audit Expert
- **Inputs**: conversation history (`<recent_messages>`), Response A (`<content_a>`), Response B (`<content_b>`)
- **Output**: strict XML — exactly one of the following two forms:

```xml
<result>Included</result>
```
```xml
<result>Not Included</result>
```

**Decision criteria:**
- `Included`: Response B covers the core technical facts, action steps, or key conclusions needed to solve the user's problem as described in Response A
- `Not Included`: Response B is missing critical operational guidance or key conclusions, or takes an entirely different technical approach

---

### 2.4 Call Interface: scoring_content

Core entry point: `cirrus.judge.scoring.scoring_content`

```python
from cirrus.judge.scoring import scoring_content

score = scoring_content(
    contentA="Reference answer text",
    contentB="Model output text to evaluate",
    history="Conversation history (can be an empty string)",
    model=None,       # optional — uses judge_config.yaml default when omitted
    llm_args=None,    # optional — uses default temperature when omitted
)
```

**Overriding the default model and parameters:**

```python
# Use a stronger model as the judge
score = scoring_content(
    contentA=ref_answer,
    contentB=model_output,
    history="",
    model="qwen3-max",
    llm_args={"temperature": 0.0},
)
```

---

### 2.5 Return Values

| Return value | Meaning |
|--------------|---------|
| `1` | LLM judged `Included` — B contains the core content of A |
| `0` | LLM judged `Not Included` — B does not contain the core content of A |
| `-1` | LLM output could not be parsed, or an exception occurred during the call |

---

### 2.6 Testing the Judge Module

File: `tests/test_judge.py`

All tests use mocks in place of real LLM calls — **no API quota is consumed** and the tests can be run at any time.

```bash
# Run all judge tests
pytest tests/test_judge.py -v
```

**Test cases included:**

| Test class | Test name | Description |
|------------|-----------|-------------|
| `TestLoadJudgeConfig` | `test_config_file_exists` | Verifies `judge_config.yaml` exists |
| | `test_load_returns_dict` | Verifies the loaded result is a dict |
| | `test_load_has_model` | Verifies the `model` field is present |
| | `test_load_has_temperature` | Verifies the `temperature` field is present |
| | `test_load_default_values` | Verifies defaults match expectations (`deepseek-chat`, `0.0`) |
| | `test_load_fallback_on_missing_keys` | Verifies safe fallback to `{}` when the `judge` key is missing |
| `TestScoringContentWithDefaultConfig` | `test_returns_1_when_included` | LLM returns `Included` → score is 1 |
| | `test_returns_0_when_not_included` | LLM returns `Not Included` → score is 0 |
| | `test_returns_minus1_on_unknown_response` | LLM returns unrecognized content → score is -1 |
| | `test_returns_minus1_on_exception` | LLM call raises an exception → score is -1 |
| | `test_uses_default_model_from_config` | Uses config default when `model` is not passed |
| | `test_uses_default_temperature_from_config` | Uses config default temperature when `llm_args` is not passed |
| | `test_override_model` | Explicit `model` overrides the config default |
| | `test_override_llm_args` | Explicit `llm_args` overrides the default temperature |

---

## 3. Adding a New Provider

**Step 1:** Add the new provider to `configs/providers_config.yaml`:

```yaml
my_provider:
  base_url: "https://api.my-provider.com/v1"
  base_url_env: "MY_PROVIDER_BASE_URL"
  api_key_env: "MY_PROVIDER_API_KEY"
  default_model: "my-model-name"
```

**Step 2:** Add the corresponding key to `.env`:

```env
MY_PROVIDER_API_KEY=your_api_key_here
```

**Step 3:** If you want automatic provider inference, add a recognition rule to `guess_provider_from_model` in `src/cirrus/llm/service.py`:

```python
elif 'my-model' in model_lower:
    return 'my_provider'
```

After these steps, you can call the new provider with `call_llm(..., provider="my_provider")`. The integration tests will automatically include it as well.
