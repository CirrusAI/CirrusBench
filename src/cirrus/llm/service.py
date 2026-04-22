# -*- coding: utf-8 -*-
"""
Unified LLM Service Module - 简化版本
支持 DeepSeek, OpenAI, Qwen, Claude 等基于 OpenAI 协议的提供商
"""

import os
import json
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger

from cirrus.llm.llm_config import LLMConfig


@dataclass
class ToolFunction:
    name: str
    arguments: str
    arguments_dict: Optional[Dict[str, Any]] = None


@dataclass
class ToolCall:
    id: str
    type: str
    function: ToolFunction


@dataclass
class Usage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class LLMResponse:
    content: Optional[str]
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Optional[Usage] = None
    finish_reason: Optional[str] = None
    raw_response: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0



def get_llm_client(provider: str = 'deepseek',
                  base_url: Optional[str] = None,
                  api_key: Optional[str] = None) -> OpenAI:
    """
    获取 LLM 客户端

    Args:
        provider: 提供商名称
        base_url: 自定义 base_url（可选）
        api_key: 自定义 api_key（可选）

    Returns:
        OpenAI: 客户端实例
    """
    load_dotenv()

    provider = provider.lower()

    if provider not in LLMConfig.PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}. Supported: {list(LLMConfig.PROVIDERS.keys())}")

    config = LLMConfig.PROVIDERS[provider]

    # 获取 API key
    if api_key is None:
        api_key = os.environ.get(config['api_key_env'])
        if not api_key:
            raise ValueError(f"API key not found. Please set {config['api_key_env']} in environment or pass api_key parameter")

    # 获取 base_url：显式参数 > 环境变量 > 配置默认值
    if base_url is None:
        base_url = os.environ.get(config.get('base_url_env', '')) or config['base_url']

    # 创建客户端
    client_kwargs = {'api_key': api_key}
    if base_url:
        client_kwargs['base_url'] = base_url

    return OpenAI(**client_kwargs)


def parse_response(response) -> LLMResponse:
    """
    解析 OpenAI 格式的响应

    Args:
        response: OpenAI 响应对象

    Returns:
        LLMResponse: 统一的响应格式
    """
    choice = response.choices[0]
    message = choice.message

    content = message.content

    # 解析 tool calls
    parsed_tool_calls: List[ToolCall] = []
    if getattr(message, "tool_calls", None):
        for tc in message.tool_calls:
            arguments_raw = tc.function.arguments
            try:
                arguments_dict = json.loads(arguments_raw) if arguments_raw else {}
            except json.JSONDecodeError:
                arguments_dict = None

            parsed_tool_calls.append(
                ToolCall(
                    id=tc.id,
                    type=tc.type,
                    function=ToolFunction(
                        name=tc.function.name,
                        arguments=arguments_raw,
                        arguments_dict=arguments_dict
                    )
                )
            )

    # 解析 usage
    usage_obj = None
    if getattr(response, "usage", None):
        usage_obj = Usage(
            input_tokens=getattr(response.usage, "prompt_tokens", None),
            output_tokens=getattr(response.usage, "completion_tokens", None),
            total_tokens=getattr(response.usage, "total_tokens", None)
        )

    return LLMResponse(
        content=content,
        tool_calls=parsed_tool_calls,
        usage=usage_obj,
        finish_reason=choice.finish_reason,
        raw_response=response
    )


def guess_provider_from_model(model: str) -> str:
    """
    根据模型名称猜测提供商

    Args:
        model: 模型名称

    Returns:
        str: 猜测的提供商名称
    """
    model_lower = model.lower()

    # 根据模型名称的常见模式来猜测提供商
    if 'deepseek' in model_lower:
        return 'deepseek'
    elif any(prefix in model_lower for prefix in ['gpt-', 'gpt4', 'o1-']):
        return 'openai'
    elif any(prefix in model_lower for prefix in ['qwen', 'qwen2']):
        return 'qwen'
    elif any(prefix in model_lower for prefix in ['claude-', 'claude']):
        return 'claude'

    # 默认返回 deepseek（可以根据需要调整默认值）
    return 'deepseek'


def call_llm(messages: List[Dict],
             model: str,
             provider: Optional[str] = None,
             tools: Optional[List[Dict]] = None,
             base_url: Optional[str] = None,
             api_key: Optional[str] = None,
             **kwargs) -> LLMResponse:
    """
    统一的 LLM 调用接口

    Args:
        messages: 对话消息列表
        model: 模型名称
        provider: 提供商名称 ('deepseek', 'openai', 'qwen', 'claude' 等)，如果为None则根据模型名称自动推断
        tools: 工具定义列表
        base_url: 自定义 API 地址
        api_key: 自定义 API 密钥
        **kwargs: 其他参数

    Returns:
        LLMResponse: 统一的响应格式
    """
    # 过滤掉非 OpenAI API 的自定义参数
    kwargs.pop("num_retries", None)

    # 如果没有提供 provider，根据模型名称进行猜测
    if provider is None:
        provider = guess_provider_from_model(model)
        logger.info(f"Auto-detected provider '{provider}' for model '{model}'")

    # 标准 OpenAI 协议服务
    client = get_llm_client(provider, base_url, api_key)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools or [],
        tool_choice="auto" if tools else None,
        stream=False,
        **kwargs
    )

    return parse_response(response)



# 便捷函数
def call_deepseek(messages: List[Dict], model: str = 'deepseek-chat', **kwargs) -> LLMResponse:
    """调用 DeepSeek"""
    return call_llm(messages, model, 'deepseek', **kwargs)


def call_openai(messages: List[Dict], model: str = 'gpt-3.5-turbo', **kwargs) -> LLMResponse:
    """调用 OpenAI"""
    return call_llm(messages, model, 'openai', **kwargs)


def call_qwen(messages: List[Dict], model: str = 'qwen-turbo', **kwargs) -> LLMResponse:
    """调用 Qwen"""
    return call_llm(messages, model, 'qwen', **kwargs)


def call_claude(messages: List[Dict], model: str = 'claude-3-sonnet-20240229', **kwargs) -> LLMResponse:
    """调用 Claude"""
    return call_llm(messages, model, 'claude', **kwargs)


if __name__ == '__main__':
    # 测试消息
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "请帮助我查询今天北京的天气情况"}
    ]

    # 测试工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名"},
                        "date": {"type": "string", "description": "日期"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    # 测试不同提供商（手动指定）
    test_cases_manual = [
        ('deepseek', 'deepseek-chat'),
        ('openai', 'gpt-3.5-turbo'),
        ('qwen', 'qwen-turbo'),
        ('claude', 'claude-3-sonnet-20240229')
    ]

    for provider, model in test_cases_manual:
        try:
            print(f"\n=== 测试 {provider} (手动指定) ===")
            resp = call_llm(
                messages=messages,
                model=model,
                provider=provider,
                tools=tools
            )

            print(f"Content: {resp.content}")
            print(f"Tool calls: {len(resp.tool_calls)}")
            print(f"Usage: {resp.usage}")

        except Exception as e:
            print(f"Error with {provider}: {str(e)}")

    # 测试自动推断提供商
    test_models_auto = [
        'deepseek-chat',
        'gpt-4o',
        'qwen-max',
        'claude-3-sonnet-20240229'
    ]

    print(f"\n=== 测试自动推断 Provider ===")
    for model in test_models_auto:
        try:
            guessed_provider = guess_provider_from_model(model)
            print(f"Model: {model} -> Provider: {guessed_provider}")

            # 实际调用（注释掉以避免实际 API 调用）
            # resp = call_llm(
            #     messages=messages,
            #     model=model,  # 不指定 provider，让函数自动推断
            #     tools=tools
            # )

        except Exception as e:
            print(f"Error with model {model}: {str(e)}")