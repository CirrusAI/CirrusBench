# 大模型调用与 Judge 配置指南

**Language / 语言：** [English](LLM_AND_JUDGE_GUIDE.md) | 中文

本文档介绍项目中大模型（LLM）调用的配置方式、使用方法，以及 Judge 评分模块的配置与测试。

---

## 目录

- [一、大模型调用](#一大模型调用)
  - [1.1 支持的提供商](#11-支持的提供商)
  - [1.2 配置文件：providers_config.yaml](#12-配置文件providers_configyaml)
  - [1.3 环境变量配置（.env）](#13-环境变量配置env)
  - [1.4 调用接口](#14-调用接口)
  - [1.5 响应数据结构](#15-响应数据结构)
  - [1.6 工具调用（Tool Call）](#16-工具调用tool-call)
  - [1.7 自动推断提供商](#17-自动推断提供商)
  - [1.8 便捷封装函数](#18-便捷封装函数)
  - [1.9 测试大模型调用](#19-测试大模型调用)
- [二、Judge 评分模块](#二judge-评分模块)
  - [2.1 功能概述](#21-功能概述)
  - [2.2 配置文件：judge_config.yaml](#22-配置文件judge_configyaml)
  - [2.3 Judge Prompt](#23-judge-prompt)
  - [2.4 调用接口：scoring_content](#24-调用接口scoring_content)
  - [2.5 返回值说明](#25-返回值说明)
  - [2.6 测试 Judge 模块](#26-测试-judge-模块)
- [三、扩展新提供商](#三扩展新提供商)

---

## 一、大模型调用

### 1.1 支持的提供商

| 提供商 | 标识符 | 协议 |
|--------|--------|------|
| DeepSeek | `deepseek` | OpenAI 兼容 |
| OpenAI | `openai` | OpenAI 原生 |
| 通义千问（Qwen） | `qwen` | OpenAI 兼容 |
| Anthropic Claude | `claude` | OpenAI 兼容代理 |

所有提供商均通过统一的 `call_llm` 接口调用，底层使用 OpenAI SDK。

---

### 1.2 配置文件：providers_config.yaml

路径：`configs/providers_config.yaml`

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

**字段说明：**

| 字段 | 说明 |
|------|------|
| `base_url` | 硬编码的默认 API 地址，作为兜底值 |
| `base_url_env` | 环境变量名称，若该变量已设置则优先使用其值覆盖 `base_url`（适用于代理或自定义接口） |
| `api_key_env` | 存放 API Key 的环境变量名称 |
| `default_model` | 该提供商的默认模型名，用于测试时自动选取 |

**base_url 与 base_url_env 的优先级：**

```
显式传参 base_url  >  环境变量 (base_url_env)  >  配置文件 base_url
```

---

### 1.3 环境变量配置（.env）

复制模板文件并填写各提供商的 API Key：

```bash
cp .env.template .env
```

编辑 `.env`：

```env
# DeepSeek
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# 通义千问（阿里云 DashScope）
QWEN_API_KEY=your_qwen_api_key_here

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# 可选：覆盖 base_url（例如使用代理或本地部署）
# DEEPSEEK_BASE_URL=http://your-proxy/v1
```

> `.env` 文件已加入 `.gitignore`，请勿提交到版本控制。

---

### 1.4 调用接口

核心入口：`cirrus.llm.service.call_llm`

```python
from cirrus.llm.service import call_llm

response = call_llm(
    messages=[{"role": "user", "content": "你好"}],
    model="deepseek-chat",
    provider="deepseek",   # 可选，不填则根据模型名自动推断
    tools=None,            # 可选，工具定义列表
    base_url=None,         # 可选，覆盖默认 base_url
    api_key=None,          # 可选，覆盖环境变量中的 key
    max_tokens=1024,       # 其余 OpenAI 兼容参数均可透传
    temperature=0.7,
)
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `messages` | `List[Dict]` | OpenAI 格式消息列表 |
| `model` | `str` | 模型名称 |
| `provider` | `str` \| `None` | 提供商标识符，`None` 时自动推断 |
| `tools` | `List[Dict]` \| `None` | Function Calling 工具定义 |
| `base_url` | `str` \| `None` | 覆盖提供商默认地址 |
| `api_key` | `str` \| `None` | 覆盖环境变量中的 API Key |
| `**kwargs` | - | 其他透传至 OpenAI SDK 的参数（如 `temperature`、`max_tokens`） |

---

### 1.5 响应数据结构

`call_llm` 返回统一的 `LLMResponse` 对象：

```python
@dataclass
class LLMResponse:
    content: Optional[str]           # 文本回复内容
    tool_calls: List[ToolCall]       # 工具调用列表（无调用时为空列表）
    usage: Optional[Usage]           # Token 用量
    finish_reason: Optional[str]     # 结束原因（"stop" / "tool_calls" / "length" 等）
    raw_response: Any                # 原始 OpenAI 响应对象

    @property
    def has_tool_calls(self) -> bool: ...  # 是否包含工具调用

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
    arguments: str                   # JSON 字符串
    arguments_dict: Optional[Dict]   # 解析后的字典
```

---

### 1.6 工具调用（Tool Call）

```python
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "date": {"type": "string", "description": "日期，格式 YYYY-MM-DD"}
            },
            "required": ["city"]
        }
    }
}

# 第一轮：让模型调用工具
resp = call_llm(
    messages=[
        {"role": "system", "content": "你是天气助手，收到天气询问时调用 get_weather 工具。"},
        {"role": "user", "content": "北京今天天气怎么样？"}
    ],
    model="deepseek-chat",
    provider="deepseek",
    tools=[WEATHER_TOOL],
)

if resp.has_tool_calls:
    tool_call = resp.tool_calls[0]
    city = tool_call.function.arguments_dict["city"]

    # 执行工具，获取结果
    weather_result = f"{city} 天气：晴，25°C"

    # 第二轮：将工具结果回传给模型
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

### 1.7 自动推断提供商

不传 `provider` 时，`call_llm` 会根据模型名自动推断：

| 模型名包含 | 推断提供商 |
|-----------|-----------|
| `deepseek` | `deepseek` |
| `gpt-`、`gpt4`、`o1-` | `openai` |
| `qwen`、`qwen2` | `qwen` |
| `claude-`、`claude` | `claude` |
| 其他 | `deepseek`（默认） |

```python
# 不指定 provider，自动推断为 openai
resp = call_llm(messages=[...], model="gpt-4o")
```

---

### 1.8 便捷封装函数

```python
from cirrus.llm.service import call_deepseek, call_openai, call_qwen, call_claude

# 等价于 call_llm(..., provider="deepseek", model="deepseek-chat")
resp = call_deepseek(messages=[...])

# 等价于 call_llm(..., provider="openai", model="gpt-3.5-turbo")
resp = call_openai(messages=[...])

resp = call_qwen(messages=[...])
resp = call_claude(messages=[...])
```

---

### 1.9 测试大模型调用

#### 集成测试（真实 API 调用）

文件：`tests/test_providers_integration.py`

该测试会读取 `providers_config.yaml` 中的所有提供商，逐一发起真实 API 调用。**未设置对应环境变量的提供商会被自动跳过**，不会导致测试失败。

```bash
# 运行所有集成测试
pytest tests/test_providers_integration.py -v -s

# 只运行某个提供商
pytest tests/test_providers_integration.py -v -s -k "deepseek"

# 只运行连通性测试
pytest tests/test_providers_integration.py -v -s -k "test_basic_connectivity"
```

**包含的测试项：**

| 测试名 | 说明 |
|--------|------|
| `test_show_providers_config` | 展示配置概览，检查哪些提供商已设置 Key（不发起 API 调用） |
| `test_basic_connectivity` | 基础对话连通性测试 |
| `test_tool_call` | 工具调用完整流程测试（含多轮对话） |
| `test_usage_info` | Token 用量信息格式验证 |
| `test_multi_turn` | 多轮对话上下文保持测试 |

**快速验证配置是否正确（无 API 调用）：**

```bash
pytest tests/test_providers_integration.py::test_show_providers_config -v -s
```

输出示例：
```
providers_config.yaml 共配置了 4 个提供商：
  [deepseek] model=deepseek-chat, key=DEEPSEEK_API_KEY (✓ 已设置)
  [openai]   model=gpt-3.5-turbo, key=OPENAI_API_KEY  (✗ 未设置)
  [qwen]     model=qwen-turbo,    key=QWEN_API_KEY    (✓ 已设置)
  [claude]   model=claude-3-sonnet-20240229, key=ANTHROPIC_API_KEY (✗ 未设置)
```

---

## 二、Judge 评分模块

### 2.1 功能概述

Judge 模块用于评估两段回复的内容覆盖关系：判断 **Response B 是否实质性包含了 Response A 的核心解决方案**，返回量化评分。

典型使用场景：在 benchmark 评测中，以参考答案作为 Response A，以待评测模型的输出作为 Response B，通过大模型打分来衡量回复质量。

---

### 2.2 配置文件：judge_config.yaml

路径：`configs/judge_config.yaml`

```yaml
judge:
  model: deepseek-chat
  temperature: 0.0
```

**字段说明：**

| 字段 | 说明 |
|------|------|
| `model` | Judge 使用的模型名，需在 `providers_config.yaml` 中有对应提供商 |
| `temperature` | 推理温度，建议设为 `0.0` 以保证输出稳定性 |

调用 `scoring_content` 时，如果不显式传入 `model` 或 `llm_args`，程序会自动读取此文件的默认值。

---

### 2.3 Judge Prompt

路径：`data/prompts/judge/judge_prompt.md`

Judge Prompt 定义了大模型扮演的角色和判断逻辑：

- **角色**：高级信息审计专家
- **输入**：历史对话（`<recent_messages>`）、Response A（`<content_a>`）、Response B（`<content_b>`）
- **输出**：严格的 XML 格式，只能是以下两种之一：

```xml
<result>Included</result>
```
```xml
<result>Not Included</result>
```

**判断标准：**
- `Included`：Response B 覆盖了 Response A 中解决用户问题的核心技术事实、操作路径或关键结论
- `Not Included`：Response B 缺少关键操作指引或核心判断结论，或走了完全不同的技术路线

---

### 2.4 调用接口：scoring_content

核心入口：`cirrus.judge.scoring.scoring_content`

```python
from cirrus.judge.scoring import scoring_content

score = scoring_content(
    contentA="参考答案文本",
    contentB="待评测回复文本",
    history="历史对话文本（可为空字符串）",
    model=None,       # 可选，不填则读取 judge_config.yaml 中的默认模型
    llm_args=None,    # 可选，不填则读取默认 temperature
)
```

**覆盖默认模型和参数示例：**

```python
# 使用更强的模型做 judge
score = scoring_content(
    contentA=ref_answer,
    contentB=model_output,
    history="",
    model="qwen3-max",
    llm_args={"temperature": 0.0},
)
```

---

### 2.5 返回值说明

| 返回值 | 含义 |
|--------|------|
| `1` | LLM 判断为 `Included`，B 包含 A 的核心内容 |
| `0` | LLM 判断为 `Not Included`，B 未包含 A 的核心内容 |
| `-1` | LLM 输出无法识别，或调用过程中发生异常 |

---

### 2.6 测试 Judge 模块

文件：`tests/test_judge.py`

该文件使用 mock 替换真实 LLM 调用，所有测试**不消耗 API 额度**，可直接运行。

```bash
# 运行所有 judge 测试
pytest tests/test_judge.py -v
```

**包含的测试项：**

| 测试类 | 测试名 | 说明 |
|--------|--------|------|
| `TestLoadJudgeConfig` | `test_config_file_exists` | 验证 `judge_config.yaml` 文件存在 |
| | `test_load_returns_dict` | 验证加载结果为字典 |
| | `test_load_has_model` | 验证包含 `model` 字段 |
| | `test_load_has_temperature` | 验证包含 `temperature` 字段 |
| | `test_load_default_values` | 验证默认值符合预期（`deepseek-chat`, `0.0`） |
| | `test_load_fallback_on_missing_keys` | 验证配置文件缺少 `judge` 节点时安全返回 `{}` |
| `TestScoringContentWithDefaultConfig` | `test_returns_1_when_included` | LLM 返回 `Included` → 得分 1 |
| | `test_returns_0_when_not_included` | LLM 返回 `Not Included` → 得分 0 |
| | `test_returns_minus1_on_unknown_response` | LLM 返回无法识别内容 → 得分 -1 |
| | `test_returns_minus1_on_exception` | LLM 调用异常 → 得分 -1 |
| | `test_uses_default_model_from_config` | 未传 `model` 时使用配置文件默认值 |
| | `test_uses_default_temperature_from_config` | 未传 `llm_args` 时使用配置文件默认温度 |
| | `test_override_model` | 显式传 `model` 时覆盖默认值 |
| | `test_override_llm_args` | 显式传 `llm_args` 时覆盖默认温度 |

---

## 三、扩展新提供商

**Step 1：** 在 `configs/providers_config.yaml` 中添加新提供商：

```yaml
my_provider:
  base_url: "https://api.my-provider.com/v1"
  base_url_env: "MY_PROVIDER_BASE_URL"
  api_key_env: "MY_PROVIDER_API_KEY"
  default_model: "my-model-name"
```

**Step 2：** 在 `.env` 中添加对应的 Key：

```env
MY_PROVIDER_API_KEY=your_api_key_here
```

**Step 3：** 如需自动推断提供商，在 `src/cirrus/llm/service.py` 的 `guess_provider_from_model` 函数中添加识别规则：

```python
elif 'my-model' in model_lower:
    return 'my_provider'
```

完成以上步骤后，即可通过 `call_llm(..., provider="my_provider")` 调用新提供商，集成测试也会自动覆盖该提供商。
