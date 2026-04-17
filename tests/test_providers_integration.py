# -*- coding: utf-8 -*-
"""
集成测试：读取 providers_config.yaml 中的所有配置模型，进行真实 API 调用测试
包括普通对话测试和工具调用测试
"""

import os
import sys
import pytest
from pathlib import Path
from dotenv import load_dotenv

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv()

from cirrus.llm.service import call_llm, LLMResponse
from cirrus.llm.llm_config import LLMConfig

# ── 测试用工具定义 ─────────────────────────────────────────────────────────────
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如：北京、上海"
                },
                "date": {
                    "type": "string",
                    "description": "日期，格式 YYYY-MM-DD，默认今天"
                }
            },
            "required": ["city"]
        }
    }
}

# ── 工具函数（模拟） ───────────────────────────────────────────────────────────
def get_weather(city: str, date: str = "today") -> str:
    return f"{city} {date} 天气：晴，25°C，东南风3级"


def get_api_key_for_provider(cfg: dict) -> str:
    """从环境变量获取指定提供商的 API key"""
    api_key_env = cfg.get("api_key_env", "")
    return os.environ.get(api_key_env, "")


def get_base_url_for_provider(cfg: dict) -> str:
    """获取提供商的 base_url（环境变量优先）"""
    base_url_env = cfg.get("base_url_env", "")
    return os.environ.get(base_url_env, "") or cfg.get("base_url", "")


# ── 参数化：从 LLMConfig 收集所有 (provider, model) 对 ──────────────────────
def collect_test_cases():
    """收集所有提供商的 (provider_name, model, api_key, base_url) 四元组"""
    cases = []
    for provider_name, cfg in LLMConfig.PROVIDERS.items():
        api_key = get_api_key_for_provider(cfg)
        base_url = get_base_url_for_provider(cfg)
        model = cfg.get("default_model", "")
        cases.append(
            pytest.param(
                provider_name,
                model,
                api_key,
                base_url,
                id=f"{provider_name}::{model}"
            )
        )
    return cases


ALL_CASES = collect_test_cases()


# ── 工具函数：跳过没有 API key 的 provider ────────────────────────────────────
def skip_if_no_key(api_key: str, provider_name: str):
    if not api_key:
        cfg = LLMConfig.PROVIDERS[provider_name]
        env_var = cfg.get("api_key_env", "")
        pytest.skip(f"跳过 {provider_name}：环境变量 {env_var} 未设置")


# ════════════════════════════════════════════════════════════════════════════════
# 测试 1：普通对话连通性测试
# ════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("provider_name,model,api_key,base_url", ALL_CASES)
def test_basic_connectivity(provider_name: str, model: str, api_key: str, base_url: str):
    """测试各提供商的基本对话连通性"""
    skip_if_no_key(api_key, provider_name)

    messages = [
        {"role": "user", "content": "你好，请用一句话介绍你自己。"}
    ]

    resp = call_llm(
        messages=messages,
        model=model,
        provider=provider_name,
        api_key=api_key,
        base_url=base_url or None,
    )

    assert isinstance(resp, LLMResponse), "返回值应为 LLMResponse"
    assert resp.content is not None and len(resp.content) > 0, "content 不应为空"
    assert not resp.has_tool_calls, "普通对话不应触发工具调用"
    assert resp.finish_reason in ("stop", "length", "end_turn", None), \
        f"非预期的 finish_reason: {resp.finish_reason}"

    print(f"\n[{provider_name}][{model}] content: {resp.content[:100]}")
    if resp.usage:
        print(f"  usage: input={resp.usage.input_tokens}, output={resp.usage.output_tokens}")


# ════════════════════════════════════════════════════════════════════════════════
# 测试 2：工具调用测试（让模型调用 get_weather）
# ════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("provider_name,model,api_key,base_url", ALL_CASES)
def test_tool_call(provider_name: str, model: str, api_key: str, base_url: str):
    """测试各提供商的工具调用能力"""
    skip_if_no_key(api_key, provider_name)

    messages = [
        {"role": "system", "content": "你是一个天气助手。当用户询问天气时，请使用 get_weather 工具获取信息。"},
        {"role": "user", "content": "请帮我查询北京今天的天气。"}
    ]

    resp = call_llm(
        messages=messages,
        model=model,
        provider=provider_name,
        api_key=api_key,
        base_url=base_url or None,
        tools=[WEATHER_TOOL],
    )

    assert isinstance(resp, LLMResponse), "返回值应为 LLMResponse"

    if resp.has_tool_calls:
        # 模型触发了工具调用
        assert len(resp.tool_calls) >= 1
        tool_call = resp.tool_calls[0]

        assert tool_call.function.name == "get_weather", \
            f"期望调用 get_weather，实际: {tool_call.function.name}"
        assert tool_call.id is not None and len(tool_call.id) > 0
        assert tool_call.function.arguments_dict is not None, "arguments 应能被解析为 dict"
        assert "city" in tool_call.function.arguments_dict, "参数中应包含 city"

        city = tool_call.function.arguments_dict["city"]
        assert len(city) > 0, "city 不应为空字符串"

        print(f"\n[{provider_name}][{model}] 工具调用: get_weather(city={city})")

        # 模拟执行工具并继续对话
        weather_result = get_weather(city)
        follow_up_messages = messages + [
            {
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": weather_result,
            }
        ]

        final_resp = call_llm(
            messages=follow_up_messages,
            model=model,
            provider=provider_name,
            api_key=api_key,
            base_url=base_url or None,
        )

        assert isinstance(final_resp, LLMResponse)
        assert final_resp.content is not None and len(final_resp.content) > 0, \
            "工具返回后的最终回复不应为空"

        print(f"  最终回复: {final_resp.content[:150]}")

    else:
        # 模型直接用文本回答（部分模型不一定触发工具调用）
        assert resp.content is not None and len(resp.content) > 0, \
            "未触发工具调用时 content 不应为空"
        print(f"\n[{provider_name}][{model}] 未触发工具调用，文本回复: {resp.content[:100]}")


# ════════════════════════════════════════════════════════════════════════════════
# 测试 3：usage 信息测试
# ════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("provider_name,model,api_key,base_url", ALL_CASES)
def test_usage_info(provider_name: str, model: str, api_key: str, base_url: str):
    """测试各提供商是否正确返回 token 用量信息"""
    skip_if_no_key(api_key, provider_name)

    messages = [{"role": "user", "content": "1+1=?"}]

    resp = call_llm(
        messages=messages,
        model=model,
        provider=provider_name,
        api_key=api_key,
        base_url=base_url or None,
    )

    assert isinstance(resp, LLMResponse)
    assert resp.content is not None

    if resp.usage is not None:
        if resp.usage.input_tokens is not None:
            assert resp.usage.input_tokens > 0, "input_tokens 应大于 0"
        if resp.usage.output_tokens is not None:
            assert resp.usage.output_tokens > 0, "output_tokens 应大于 0"
        if resp.usage.total_tokens is not None and resp.usage.input_tokens is not None \
                and resp.usage.output_tokens is not None:
            assert resp.usage.total_tokens == resp.usage.input_tokens + resp.usage.output_tokens, \
                "total_tokens 应等于 input + output"

        print(f"\n[{provider_name}][{model}] usage: {resp.usage}")
    else:
        print(f"\n[{provider_name}][{model}] 未返回 usage 信息（可能正常）")


# ════════════════════════════════════════════════════════════════════════════════
# 测试 4：多轮对话测试
# ════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("provider_name,model,api_key,base_url", ALL_CASES)
def test_multi_turn(provider_name: str, model: str, api_key: str, base_url: str):
    """测试多轮对话上下文保持"""
    skip_if_no_key(api_key, provider_name)

    # 第一轮
    messages = [
        {"role": "user", "content": "我的名字叫小明，请记住。"}
    ]
    resp1 = call_llm(
        messages=messages,
        model=model,
        provider=provider_name,
        api_key=api_key,
        base_url=base_url or None,
    )
    assert resp1.content is not None

    # 第二轮（测试上下文是否保持）
    messages.append({"role": "assistant", "content": resp1.content})
    messages.append({"role": "user", "content": "我叫什么名字？"})

    resp2 = call_llm(
        messages=messages,
        model=model,
        provider=provider_name,
        api_key=api_key,
        base_url=base_url or None,
    )

    assert resp2.content is not None and len(resp2.content) > 0
    assert "小明" in resp2.content, \
        f"期望第二轮回复中包含「小明」，实际: {resp2.content}"

    print(f"\n[{provider_name}][{model}] 多轮对话回复: {resp2.content}")


# ════════════════════════════════════════════════════════════════════════════════
# 非参数化：快速打印配置概览（不做真实调用）
# ════════════════════════════════════════════════════════════════════════════════
def test_show_providers_config():
    """展示 providers_config.yaml 中配置的所有提供商"""
    providers = LLMConfig.PROVIDERS
    print(f"\n\nproviders_config.yaml 共配置了 {len(providers)} 个提供商：")
    for name, cfg in providers.items():
        model = cfg.get("default_model", "N/A")
        env_var = cfg.get("api_key_env", "N/A")
        key_set = "✓ 已设置" if os.environ.get(env_var) else "✗ 未设置"
        print(f"  [{name}] model={model}, key={env_var} ({key_set})")

    assert len(providers) > 0, "providers_config.yaml 应至少配置一个提供商"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
