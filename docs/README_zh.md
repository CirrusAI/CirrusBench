# Cirrus — LLM Agent 基准测试框架

**Language / 语言：** [English](../README.md) | 中文

Cirrus 是一个模块化的 LLM 智能体基准测试框架，用于在真实客服仿真场景中评估大语言模型的表现。支持多轮对话、工具调用任务、多提供商 LLM 接入和自动评分。

---

## 目录

- [项目概述](#项目概述)
- [架构说明](#架构说明)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [运行基准测试](#运行基准测试)
- [评估与指标](#评估与指标)
- [支持的 LLM 提供商](#支持的-llm-提供商)
- [项目结构](#项目结构)
- [许可证](#许可证)

---

## 项目概述

Cirrus 在两个任务域上评估 LLM 智能体：

| 域 | 说明 |
|----|------|
| `no_tool` | 纯对话任务 — 智能体仅通过对话处理用户请求 |
| `with_tool` | 工具调用任务 — 智能体可调用外部工具（API、数据库查询等） |

每个任务进行多次独立 trial，结果自动评分并汇总为标准基准指标。

---

## 架构说明

```
Orchestrator（编排器）
├── Agent（LLM 智能体）        ← 被评估的模型
├── Environment（环境）        ← 管理工具执行和状态
└── Evaluator（评估器）        ← 对每轮及最终结果打分
```

核心模块：

| 模块 | 路径 | 功能 |
|------|------|------|
| Orchestrator | `src/cirrus/orchestrator/` | 驱动仿真循环 |
| LLM Service | `src/cirrus/llm/` | 统一多提供商 LLM 接口 |
| Agent | `src/cirrus/agent/` | LLM 智能体实现 |
| Environment | `src/cirrus/environment/` | 工具执行沙箱 |
| Evaluator | `src/cirrus/evaluation/` | 逐轮与最终评分 |
| Judge | `src/cirrus/judge/` | LLM-as-judge 内容评分 |
| Metrics | `src/cirrus/metrics/` | pass^k 与 reward 汇总 |
| Data Models | `src/cirrus/data_model/` | 任务、消息、工具的 Pydantic 数据模型 |

---

## 快速开始

### 环境要求

- Python 3.10+
- Linux / macOS / Windows

### 安装步骤

```bash
# 1. 克隆仓库
git clone <repository-url>
cd benchmark_public

# 2. 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. 安装依赖
pip install -r requirements.txt
pip install -e .

# 4. 配置 API 密钥
cp .env.template .env
# 编辑 .env，填入所需密钥
```

---

## 配置说明

### API 密钥（`.env`）

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key
QWEN_API_KEY=your_qwen_api_key          # 阿里云 DashScope
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

只需填写实际使用的提供商对应的密钥。

### 模型配置（`configs/providers_config.yaml`）

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

有关 LLM 提供商配置和 Judge 详细说明，请参阅 [LLM_AND_JUDGE_GUIDE_zh.md](../docs/LLM_AND_JUDGE_GUIDE_zh.md)。

---

## 运行基准测试

```bash
python -m cirrus.run \
  --domain no_tool \
  --model-name deepseek-chat \
  --num-tasks 10 \
  --num-trials 3 \
  --max-steps 20 \
  --max-errors 10 \
  --max-concurrency 3 \
  --save-to outputs/simulations
```

### 主要 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--domain` | `no_tool` | 任务域：`no_tool` 或 `with_tool` |
| `--model-name` | `deepseek-chat` | 待评估的 LLM 模型 |
| `--task-ids` | — | 运行指定任务 ID，如 `--task-ids 0008 0009` |
| `--num-tasks` | — | 只运行前 N 个任务 |
| `--num-trials` | `3` | 每个任务的独立 trial 次数 |
| `--max-steps` | `20` | 每次 trial 的最大仿真步数 |
| `--max-errors` | `10` | 连续错误上限，超出则终止当前 trial |
| `--max-concurrency` | `3` | 最大并发仿真数 |
| `--save-to` | — | 结果保存路径 |
| `--overwrite` | `false` | 是否覆盖已有结果 |

---

## 评估与指标

运行结束后，计算指标：

```bash
python -m cirrus.metrics.agent_metrics --results outputs/results.json
```

### 指标说明

| 指标 | 说明 |
|------|------|
| `avg_reward` | 所有 trial 的平均奖励（1.0 = 完全成功） |
| `pass^k` | n 次 trial 中至少 k 次成功的概率（参考 [pass^k 论文](https://arxiv.org/pdf/2406.12045)） |
| `avg_agent_cost` | 每次 trial 的平均 token 消耗 |

**Judge 模块**采用 LLM-as-judge 方式，对每轮内容的流畅度和任务完成度进行评分。

---

## 支持的 LLM 提供商

| 提供商 | 示例模型 | 工具调用 |
|--------|---------|:--------:|
| DeepSeek | `deepseek-chat`、`deepseek-reasoner` | ✅ |
| OpenAI | `gpt-4o`、`gpt-4o-mini`、`gpt-4-turbo` | ✅ |
| 通义千问（DashScope） | `qwen-turbo`、`qwen-plus`、`qwen-max` | ✅ |
| Anthropic Claude | `claude-3-haiku`、`claude-3-5-sonnet` | ✅ |
| Google Gemini | `gemini-pro`、`gemini-1.5-pro` | ⚠️ 部分支持 |

如需接入新的提供商，继承 `src/cirrus/llm/providers/base.py` 中的 `BaseLLMProvider` 并在 LLM registry 中注册即可。

---

## 项目结构

```
benchmark_public/
├── configs/
│   ├── providers_config.yaml  # LLM 提供商配置
│   ├── judge_config.yaml      # Judge 模型配置
│   └── models.yaml            # 模型定义
├── data/
│   ├── prompts/               # 智能体、judge、用户模拟器的系统提示
│   └── raw_data/              # 任务数据集（.jsonl）
├── outputs/                   # 仿真结果
├── scripts/                   # 辅助脚本
├── src/cirrus/
│   ├── agent/                 # LLM 智能体
│   ├── configs/               # 路径与运行时配置
│   ├── data_model/            # Pydantic 数据模型
│   ├── environment/           # 工具执行环境
│   ├── evaluation/            # 逐轮评估器
│   ├── judge/                 # LLM-as-judge 评分
│   ├── llm/                   # 多提供商 LLM 服务
│   ├── metrics/               # 基准指标计算
│   ├── orchestrator/          # 仿真编排器
│   ├── utils/                 # 通用工具函数
│   └── run.py                 # 入口
├── tests/
├── .env.template
├── pyproject.toml
└── requirements.txt
```

---

## 许可证

本项目采用 MIT 许可证。

> **注意**：请确保 `.env` 文件不被提交到版本控制系统，切勿泄露 API 密钥。
