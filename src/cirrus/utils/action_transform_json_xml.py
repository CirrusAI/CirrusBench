"""任务：将XML格式的action转换为JSON格式的action，或者将JSON格式的action转换为XML格式的action。注意字段名称的差异。

## 核心功能

### 1. XML自动修复机制
- 函数：`auto_fix_unclosed_xml_tags(xml_str: str) -> tuple[str, bool]`
- 特点：自动检测并修复末尾缺失的XML结束标签
- 应用场景：处理不完整的XML字符串（如LLM输出被截断）
- 工作原理：
  * 维护标签栈，追踪所有打开的标签
  * 扫描完成后，按照LIFO顺序补充缺失的结束标签
  * 只修复末尾缺失的标签，不修复中间格式错误
- 示例：
  ```
  输入: "<thinking>...</thinking><action name='reply'><content>text"
  输出: "<thinking>...</thinking><action name='reply'><content>text</content></action>"
  ```

### 2. 转义策略说明

本模块提供两种转义策略：

#### 2.1 标准模式（默认）
- 函数：`action_json_dict_to_xml()` / `action_xml_to_json_dict()`
- 特点：符合XML/HTML规范，HTML实体会被转换为实际字符
- 性能：优秀，适合大多数场景
- 示例：`&nbsp;` → `\xa0`, `&copy;` → `©`

#### 2.2 HTML实体保护模式
- 函数：`enhanced_action_json_dict_to_xml(..., preserve_html_entities=True)`
         `enhanced_action_xml_to_json_dict(..., preserve_html_entities=True)`
- 特点：保持HTML实体的原始形式，避免二次转义
- 性能：略低，但提供更精确的内容控制
- 示例：`&nbsp;` → `&nbsp;` (保持不变)

### 使用建议
- 大多数场景使用标准模式即可
- 需要保持HTML实体格式时使用保护模式
- HTML实体转换是符合规范的正常行为，不是错误
- 默认启用XML自动修复（auto_fix=True），可手动禁用

格式1 - 回复客户：

<thinking>
(分析现状和参考信息，确定当前是否已明确用户问题、或已经具备哪些信息，哪些信息缺失，或者为了调用某些工具缺少什么信息等，并决策应该回复用户什么类型的内容)
</thinking>
<action name="reply">
    <goal>
    (发送这条消息的目的)
    </goal>
    <types>
    ["standard", "solution"](回复类型, List[str]，从以下选择一个或多个：standard、solution、clarify、inquire、other、appease)
    </types>
    <content>
    (消息的具体内容，客户可看到)
    </content>
</action>

旧版本的JSON格式：
```json
{
    "thinking": "分析现状和参考信息，确定当前是否已明确用户问题、或已经具备哪些信息，哪些信息缺失，或者为了调用某些工具缺少什么信息等，并决策应该回复用户什么类型的内容",
    "action_details": {
        "action": "reply",
        "goal": "发送这条消息的目的",
        "types": ["standard", "solution"],
        "content": "消息的具体内容，客户可看到"
    }
}
```

格式2 - 调用工具：
<thinking>
(分析现状和参考信息，确定当前是否已明确用户问题、或已经具备哪些信息，哪些信息缺失，或者为了调用某些工具缺少什么信息等，并决策应该回复用户什么类型的内容，并确保回复内容符合沟通规范（比如礼貌、简洁、人性化、口语化）)
</thinking>
<action name="call_tool">
<goal>
(用一句话总结为什么要调用工具，以及调用这个（些）工具对于解决客户的问题有什么样的帮助)
</goal>
<toolcalls>
    <invoke name="tool_name">
        <id>tool_id</id>
        <parameter name="param_name1">param_value1</parameter>
        ... # 可能需要传入多个参数，依据实际工具的参数来填写
    </invoke>
    ...  # 可以调用多个工具，大部分时候，一个工具调用就足够了
</toolcalls>
</action>

旧版本的JSON格式：
```json
{
    "thinking": "分析现状和参考信息，确定当前是否已明确用户问题、或已经具备哪些信息，哪些信息缺失，或者为了调用某些工具缺少什么信息等，并决策应该回复用户什么类型的内容，并确保回复内容符合沟通规范（比如礼貌、简洁、人性化、口语化）",
    "action_details": {
        "action": "call_tool",
        "goal": "用一句话总结为什么要调用工具，以及调用这个（些）工具对于解决客户的问题有什么样的帮助",
        "calls": [
            {
                "tool_name": "tool_name",
                "tool_id": "tool_id",
                "parameters": {
                    "param_name1": "param_value1",
                    "param_name2": "param_value2"
                }
            }
        ]
    }
}
```

格式3 - 转交专家：
<thinking>
(分析现状，发现当前状况在业务经验里明确指出需要执行ask_for_help)
</thinking>
<action name="ask_for_help">
    <content>
    (描述转交其他专家的原因)
    </content>
</action>

旧版本的JSON格式：
```json
{
    "thinking": "分析现状，发现当前状况在业务经验里明确指出需要执行ask_for_help",
    "action_details": {
        "action": "ask_for_help",
        "content": "描述转交其他专家的原因"
    }
}
```

"""

import json
import logging
import re
from typing import Dict, Any, List, Union

import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent
sys.path.append(str(HERE))

from xml_parser import parse_xml_str  #, escape_xml_characters

from cirrus.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
    ToolCall
)

def parse_types_string(types_str: str) -> List[str]:
    """
    解析types字符串为列表格式
    
    支持的格式：
    - "standard, solution" → ["standard", "solution"]
    - "['standard', 'solution']" → ["standard", "solution"]
    - "standard" → ["standard"]
    
    Args:
        types_str: types字符串
        
    Returns:
        解析后的类型列表
    """
    if not isinstance(types_str, str):
        return []
    
    types_str = types_str.strip()
    
    if not types_str:
        return []
    
    # 处理已经是JSON列表格式的情况
    if types_str.startswith('[') and types_str.endswith(']'):
        try:
            return json.loads(types_str)
        except json.JSONDecodeError:
            # 尝试处理单引号格式
            try:
                # 将单引号替换为双引号
                fixed_str = types_str.replace("'", '"')
                return json.loads(fixed_str)
            except json.JSONDecodeError:
                pass
    
    # 处理逗号分隔的格式
    if ',' in types_str:
        # 分割并清理每个项目
        types_list = []
        for item in types_str.split(','):
            cleaned_item = item.strip().strip('"\'')
            if cleaned_item:
                types_list.append(cleaned_item)
        return types_list
    
    # 单个类型
    return [types_str.strip().strip('"\'')]


def format_types_for_xml(types_list: List[str]) -> str:
    """
    将types列表格式化为XML格式（合法JSON字符串）
    
    Args:
        types_list: 类型列表
        
    Returns:
        格式化后的合法JSON字符串或空字符串
    """
    if not types_list:
        return ""
    
    # 输出合法的JSON字符串格式
    return json.dumps(types_list, ensure_ascii=False)


def smart_escape_xml_characters(text: str) -> str:
    """
    智能XML转义 - 识别已有的HTML实体，避免二次转义
    
    Args:
        text: 需要转义的文本
        
    Returns:
        智能转义后的文本
    """
    if not isinstance(text, str):
        return text
    
    # HTML实体的正则模式
    html_entity_pattern = r'&(?:[a-zA-Z][a-zA-Z0-9]+|#(?:x[0-9A-Fa-f]+|[0-9]+));'
    
    result = []
    i = 0
    
    while i < len(text):
        # 检查当前位置是否是HTML实体的开始
        if text[i] == '&':
            # 寻找可能的HTML实体
            remaining = text[i:]
            match = re.match(html_entity_pattern, remaining)
            
            if match:
                # 这是一个有效的HTML实体，保持原样
                entity = match.group(0)
                result.append(entity)
                i += len(entity)
                continue
            else:
                # 这是一个普通的&符号，需要转义
                result.append('&amp;')
                i += 1
                continue
        
        # 处理其他特殊字符
        char = text[i]
        if char == '<':
            result.append('&lt;')
        elif char == '>':
            result.append('&gt;')
        elif char == '"':
            result.append('&quot;')
        elif char == "'":
            result.append('&apos;')
        else:
            # 检查是否为控制字符
            char_code = ord(char)
            if (char_code == 0x09 or char_code == 0x0A or char_code == 0x0D or
                (0x20 <= char_code <= 0xD7FF) or
                (0xE000 <= char_code <= 0xFFFD) or
                (0x10000 <= char_code <= 0x10FFFF)):
                result.append(char)
            else:
                result.append(f'&#x{char_code:X};')
        
        i += 1
    
    return ''.join(result)


def minimal_unescape_xml_characters(text: str) -> str:
    """
    最小化XML反转义 - 只反转义基本XML字符，保留HTML实体
    
    Args:
        text: 需要反转义的文本
        
    Returns:
        最小化反转义后的文本
    """
    if not isinstance(text, str):
        return text
    
    # 只反转义基本的XML预定义实体
    # 注意：必须最后处理&amp;，因为其他实体包含&字符
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&apos;', "'")
    text = text.replace('&amp;', '&')    # 必须最后处理
    
    return text


def auto_fix_unclosed_xml_tags(xml_str: str) -> tuple[str, bool]:
    """
    自动修复XML字符串末尾缺失的结束标签
    
    工作原理：
    1. 扫描XML字符串，维护一个标签栈
    2. 遇到开始标签压栈，遇到结束标签弹栈
    3. 扫描结束后，栈中剩余的标签即为未关闭的标签
    4. 按照LIFO顺序在字符串末尾补上缺失的结束标签
    
    Args:
        xml_str: 可能不完整的XML字符串
        
    Returns:
        (修复后的XML字符串, 是否进行了修复)
    """
    if not xml_str or not isinstance(xml_str, str):
        return xml_str, False
    
    # 标签栈，存储 (标签名, 原始标签字符串) 元组
    tag_stack = []
    
    # 正则模式
    # 匹配开始标签（包括带属性的）
    start_tag_pattern = r'<([a-zA-Z_][\w\-]*)((?:\s+[^>]*)?)>'
    # 匹配结束标签
    end_tag_pattern = r'</([a-zA-Z_][\w\-]*)>'
    # 匹配自闭合标签
    self_closing_pattern = r'<([a-zA-Z_][\w\-]*)[^>]*/>'
    
    # 合并所有模式，按顺序尝试匹配
    combined_pattern = f'({self_closing_pattern}|{end_tag_pattern}|{start_tag_pattern})'
    
    pos = 0
    while pos < len(xml_str):
        # 从当前位置开始查找下一个标签
        match = re.search(combined_pattern, xml_str[pos:])
        if not match:
            break
        
        full_match = match.group(0)
        
        # 判断标签类型
        if full_match.endswith('/>'):
            # 自闭合标签，无需处理
            pass
        elif full_match.startswith('</'):
            # 结束标签，从栈中弹出对应的开始标签
            tag_name_match = re.match(end_tag_pattern, full_match)
            if tag_name_match:
                end_tag_name = tag_name_match.group(1)
                
                # 从栈顶查找匹配的开始标签
                if tag_stack and tag_stack[-1][0] == end_tag_name:
                    tag_stack.pop()
                else:
                    # 标签不匹配，可能是XML格式错误
                    # 这里选择忽略，继续处理
                    logging.debug(
                        f"警告: 发现不匹配的结束标签 </{end_tag_name}>, "
                        f"当前栈顶: {tag_stack[-1][0] if tag_stack else 'empty'}"
                    )
        else:
            # 开始标签，压入栈
            tag_name_match = re.match(start_tag_pattern, full_match)
            if tag_name_match:
                tag_name = tag_name_match.group(1)
                tag_stack.append((tag_name, full_match))
        
        # 移动位置
        pos += match.start() + len(full_match)
    
    # 检查是否有未关闭的标签
    if not tag_stack:
        return xml_str, False
    
    # 构建修复后的XML
    fixed_xml = xml_str
    closing_tags = []
    
    # 按照LIFO顺序添加关闭标签
    while tag_stack:
        tag_name, _ = tag_stack.pop()
        closing_tags.append(f'</{tag_name}>')
    
    fixed_xml = xml_str + '\n' + '\n'.join(closing_tags)
    
    logging.info(
        f"XML自动修复: 补充了 {len(closing_tags)} 个结束标签: "
        f"{', '.join(closing_tags)}"
    )
    
    return fixed_xml, True


def get_allowed_tags_for_action(action_type: str) -> List[str]:
    """
    根据action类型获取允许的子标签列表
    
    Args:
        action_type: action类型 (reply, call_tool, ask_for_help)
        
    Returns:
        允许的子标签列表
    """
    if action_type == "reply":
        # reply类型只允许特定标签，避免content中的嵌套XML被解析
        return ["goal", "types", "content"]
    elif action_type == "call_tool":
        # call_tool可以解析所有子标签，因为需要处理复杂的工具调用结构
        return None  # None表示解析所有标签
    elif action_type == "ask_for_help":
        # ask_for_help只需要content标签
        return ["content"]
    else:
        # 未知类型，保守起见只解析基本标签
        return ["goal", "content"]



def build_action_details_reply(action_data: dict) -> dict:
    action_details = {
        "action": "reply"
    }

    # 回复客户格式
    goal_content = action_data.get("goal", "")
    if isinstance(goal_content, dict) and "#text" in goal_content:
        goal_content = goal_content["#text"]
    action_details["goal"] = goal_content
    
    types_content = action_data.get("types", [])
    if isinstance(types_content, dict) and "#text" in types_content:
        types_content = types_content["#text"]
    
    # 将types字符串解析为列表
    if isinstance(types_content, str):
        types_list = parse_types_string(types_content)
    elif isinstance(types_content, list):
        types_list = types_content
    else:
        types_list = []
    
    action_details["types"] = types_list
    
    content_val = action_data.get("content", "")
    if isinstance(content_val, dict) and "#text" in content_val:
        content_val = content_val["#text"]
    action_details["content"] = content_val

    return action_details


def build_action_details_call_tool(action_data: dict) -> dict:
    action_details = {
        "action": "call_tool"
    }
    # 调用工具格式 - 处理所有解析到的字段
        
    # 处理goal字段
    goal_content = action_data.get("goal", "")
    if isinstance(goal_content, dict) and "#text" in goal_content:
        goal_content = goal_content["#text"]
    action_details["goal"] = goal_content
    
    # 处理toolcalls字段
    toolcalls_data = action_data.get("toolcalls", {})
    invoke_data = toolcalls_data.get("invoke", [])
    
    # 确保invoke_data是列表
    if not isinstance(invoke_data, list):
        invoke_data = [invoke_data] if invoke_data else []
    
    calls = []
    for invoke in invoke_data:
        if isinstance(invoke, dict):
            call_item = {
                "tool_name": invoke.get("@name") or invoke.get("name", ""),
                "tool_id": invoke.get("id", ""),
                "parameters": {}
            }
            
            # 处理参数
            parameters = invoke.get("parameter", [])
            if not isinstance(parameters, list):
                parameters = [parameters] if parameters else []
            
            for param in parameters:
                if isinstance(param, dict):
                    param_name = param.get("@name") or param.get("name", "")
                    param_value = param.get("#text") or param.get("content", "")
                    if param_name:
                        call_item["parameters"][param_name] = param_value
            
            calls.append(call_item)
    
    action_details["calls"] = calls
    
    # call_tool类型允许解析所有子标签，所以添加其他所有字段
    for key, value in action_data.items():
        # 跳过已处理的字段和系统字段
        if key not in ["goal", "toolcalls", "@name", "name", "#text"] and not key.startswith("__"):
            # 处理可能的#text嵌套结构
            if isinstance(value, dict) and "#text" in value:
                processed_value = value["#text"]
            else:
                processed_value = value
            
            action_details[key] = processed_value
    
    return action_details


def build_action_details_ask_for_help(action_data: dict) -> dict:
    action_details = {
        "action": "ask_for_help"
    }
    # 转交专家格式
    content_val = action_data.get("content", "")
    if isinstance(content_val, dict) and "#text" in content_val:
        content_val = content_val["#text"]
    action_details["content"] = content_val
    
    return action_details



def action_xml_to_json_dict(
        action_xml: str, 
        safe_mode: bool = True,
        auto_fix: bool = False  # 1110改成了false，之前是true
) -> dict:
    """
    将XML格式的action转换为JSON字典格式
    根据不同的action类型使用不同的标签限制规则
    
    Args:
        action_xml: XML格式的action字符串
        safe_mode: 是否在错误时返回错误信息，而不是抛出异常
        auto_fix: 是否自动修复末尾缺失的XML结束标签
        
    Returns:
        JSON字典格式的action结果
    """
    try:
        # 第0步：自动修复缺失的结束标签（如果启用）
        if auto_fix:
            action_xml, was_fixed = auto_fix_unclosed_xml_tags(action_xml)
            if was_fixed:
                logging.info("已自动修复XML格式缺陷")
        
        # 第一步：解析顶层结构，获取thinking和action
        parsed_result, msg = parse_xml_str(
            content=action_xml,
            restrict_tags=["thinking", "action"],
            auto_types=["list"]
        )
        
        logging.info(f"XML解析消息: {msg}")
        
        if "error" in parsed_result:
            return {
                "error": f"XML解析失败: {parsed_result['error']}",
                "original_xml": action_xml
            }
        
        # 提取thinking和action内容
        thinking = parsed_result.get("thinking", "")
        action_data = parsed_result.get("action", {})
        
        if not action_data:
            return {
                "error": "未找到action标签",
                "parsed_result": parsed_result
            }
        
        # 获取action类型
        action_type = action_data.get("@name") or action_data.get("name", "")
        
        if not action_type:
            return {
                "error": "action标签缺少name属性",
                "action_data": action_data
            }
        
        # 构建JSON格式
        json_result = {
            "thinking": thinking,
            "action_details": {
                "action": action_type
            }
        }
        
        # 第二步：根据action类型使用不同的标签限制规则解析内部结构
        if "#text" in action_data and isinstance(action_data["#text"], str):
            # 获取该action类型允许的子标签
            allowed_tags = get_allowed_tags_for_action(action_type)
            
            # 解析action内部的子标签，使用特定的标签限制
            sub_parsed, _ = parse_xml_str(
                content=action_data["#text"],
                restrict_tags=allowed_tags,  # 根据action类型限制标签
                auto_types=["list"]
            )
            
            # 合并解析结果，保留属性
            for key, value in sub_parsed.items():
                if not key.startswith("__"):  # 跳过系统字段
                    action_data[key] = value
        
        # 根据action类型处理不同的字段
        if action_type == "reply":
            # 回复客户格式
            action_details = build_action_details_reply(action_data)
            
        elif action_type == "call_tool":
            # 调用工具格式 - 处理所有解析到的字段
            action_details = build_action_details_call_tool(action_data)
            
        elif action_type == "ask_for_help":
            # 转交专家格式
            action_details = build_action_details_ask_for_help(action_data)
            
        json_result["action_details"] = action_details
        return json_result
        
    except Exception as e:
        if safe_mode:
            logging.error(f"XML转JSON转换失败: {str(e)}")
            return {
                "error": f"转换过程中发生错误: {str(e)}",
                "original_xml": action_xml
            }
        else:
            raise e

def action_json_dict_to_message(json_dict):
    action_details = json_dict['action_details']
    if action_details['action'] == 'call_tool':
        content = ''
        raw_tool_call_list = action_details['calls']

        tool_calls = [ToolCall(name =raw_tool_call['tool_name'],arguments = raw_tool_call['parameters'] ) for raw_tool_call in raw_tool_call_list]

        message = AssistantMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls

        )
        return message
    elif action_details['action'] == 'reply':
        content = action_details['content']
        message = AssistantMessage(
            role = 'assistant',
            content = content
        )
        return message
        
def action_xml_to_message(xml):
    json_dict = action_xml_to_json_dict(xml)
    action_details = json_dict['action_details']
    if action_details['action'] == 'call_tool':
        content = None
        raw_tool_call_list = action_details['calls']

        tool_calls = [ToolCall(name =raw_tool_call['tool_name'],arguments = raw_tool_call['parameters'] ) for raw_tool_call in raw_tool_call_list]
        tool_calls = [tc.model_dump() for tc in tool_calls]
        message = AssistantMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls

        )
        return message
    elif action_details['action'] == 'reply':
        content = action_details['content']
        message = AssistantMessage(
            role = 'assistant',
            content = content,
        )
        return message



def build_xml_parts_reply(action_details: dict, escape_xml_characters=None) -> list:
    if escape_xml_characters is None:
        escape_xml_characters = lambda x: x
    xml_parts = []
    goal = escape_xml_characters(action_details.get("goal", ""))
    types_value = action_details.get("types", [])
    content = escape_xml_characters(action_details.get("content", ""))
    
    xml_parts.append("    <goal>")
    xml_parts.append(f"    {goal}")
    xml_parts.append("    </goal>")
    
    xml_parts.append("    <types>")
    if isinstance(types_value, list):
        # 保持列表格式，便于后续解析
        types_str = format_types_for_xml(types_value)
    else:
        types_str = str(types_value)
    xml_parts.append(f"    {escape_xml_characters(types_str)}")
    xml_parts.append("    </types>")
    
    xml_parts.append("    <content>")
    xml_parts.append(f"    {content}")
    xml_parts.append("    </content>")
    return xml_parts


def build_xml_parts_call_tool(action_details: dict, escape_xml_characters=None) -> list:
    if escape_xml_characters is None:
        escape_xml_characters = lambda x: x
    xml_parts = []
    
    # 处理goal字段
    goal = escape_xml_characters(action_details.get("goal", ""))
    xml_parts.append("<goal>")
    xml_parts.append(goal)
    xml_parts.append("</goal>")
    
    # 处理toolcalls字段
    xml_parts.append("<toolcalls>")
    
    calls = action_details.get("calls", [])
    for call in calls:
        tool_name = call.get("tool_name", "")
        tool_id = call.get("tool_id", "")
        parameters = call.get("parameters", {})
        
        xml_parts.append(f'    <invoke name="{tool_name}">')
        
        if tool_id:
            xml_parts.append(f"        <id>{escape_xml_characters(tool_id)}</id>")
        
        for param_name, param_value in parameters.items():
            escaped_value = escape_xml_characters(str(param_value))
            xml_parts.append(f'        <parameter name="{param_name}">{escaped_value}</parameter>')
        
        xml_parts.append("    </invoke>")
    
    xml_parts.append("</toolcalls>")
    
    # 处理其他所有字段（call_tool类型允许额外字段）
    for key, value in action_details.items():
        # 跳过已处理的字段和系统字段
        if key not in ["action", "goal", "calls"] and not key.startswith("__"):
            escaped_value = escape_xml_characters(str(value))
            xml_parts.append(f"<{key}>")
            xml_parts.append(escaped_value)
            xml_parts.append(f"</{key}>")
    
    return xml_parts


def build_xml_parts_ask_for_help(action_details: dict, escape_xml_characters=None) -> list:
    if escape_xml_characters is None:
        escape_xml_characters = lambda x: x
    xml_parts = []
    content = escape_xml_characters(action_details.get("content", ""))
    xml_parts.append("    <content>")
    xml_parts.append(f"    {content}")
    xml_parts.append("    </content>")
    return xml_parts


def action_json_dict_to_xml(
    thinking: str,
    action_details: dict,
    escape_xml_characters=None
) -> str:
    """
    将JSON字典格式的action转换为XML格式
    
    Args:
        thinking: 思考内容
        action_details: action详情字典
        
    Returns:
        XML格式的action字符串
    """
    if escape_xml_characters is None:
        escape_xml_characters = lambda x: x
    try:
        # 转义thinking内容
        escaped_thinking = escape_xml_characters(thinking)
        
        # 获取action类型
        action_type = action_details.get("action", "")
        
        if not action_type:
            raise ValueError("action_details中缺少action字段")
        
        # 构建XML
        xml_parts = []
        xml_parts.append("<thinking>")
        xml_parts.append(escaped_thinking)
        xml_parts.append("</thinking>")
        
        xml_parts.append(f'<action name="{action_type}">')
        
        if action_type == "reply":
            # 回复客户格式
            xml_parts.extend(build_xml_parts_reply(action_details))
            
        elif action_type == "call_tool":
            # 调用工具格式
            xml_parts.extend(build_xml_parts_call_tool(action_details))
            
        elif action_type == "ask_for_help":
            # 转交专家格式
            xml_parts.extend(build_xml_parts_ask_for_help(action_details))
        
        xml_parts.append("</action>")
        
        return "\n".join(xml_parts)
        
    except Exception as e:
        logging.error(f"JSON转XML转换失败: {str(e)}")
        raise ValueError(f"转换过程中发生错误: {str(e)}")


def enhanced_action_json_dict_to_xml(
        thinking: str, 
        action_details: dict, 
        preserve_html_entities: bool = False,
) -> str:
    """
    增强版JSON转XML，支持HTML实体保护
    
    Args:
        thinking: 思考内容
        action_details: action详情
        preserve_html_entities: 是否保护HTML实体不被二次转义

    Returns:
        XML格式字符串
    """
    if preserve_html_entities:
        # 使用智能转义
        # import xml_parser
        # original_escape = xml_parser.escape_xml_characters
        # xml_parser.escape_xml_characters = smart_escape_xml_characters
        result = action_json_dict_to_xml(thinking, action_details, escape_xml_characters=smart_escape_xml_characters)
        return result

    else:
        # 使用标准转义
        return action_json_dict_to_xml(thinking, action_details)


def enhanced_action_xml_to_json_dict(
        action_xml: str,                            
        preserve_html_entities: bool = False,
        safe_mode: bool = True,
        auto_fix: bool = False  # 1110改成了false，之前是true
) -> dict:
    """
    增强版XML转JSON，支持HTML实体保护和自动修复
    
    Args:
        action_xml: XML格式字符串
        preserve_html_entities: 是否保护HTML实体
        safe_mode: 是否在错误时返回错误信息，而不是抛出异常
        auto_fix: 是否自动修复末尾缺失的XML结束标签
    Returns:
        JSON字典
    """
    if preserve_html_entities:
        # 使用最小反转义
        import xml_parser
        original_unescape = xml_parser.unescape_xml_characters
        xml_parser.unescape_xml_characters = minimal_unescape_xml_characters
        
        try:
            result = action_xml_to_json_dict(
                action_xml, 
                safe_mode=safe_mode, 
                auto_fix=auto_fix
            )
            return result
        finally:
            # 恢复原始函数
            xml_parser.unescape_xml_characters = original_unescape
    else:
        # 使用标准反转义
        return action_xml_to_json_dict(
            action_xml, 
            safe_mode=safe_mode, 
            auto_fix=auto_fix
        )


def __test__():
    """
    完整的测试函数，覆盖所有格式转换场景，包含期望结果对比
    注意：这是一个简化版本，完整的测试在 /tmp_scripts/improved_action_transform_test.py
    """
    print("=" * 80)
    print("开始测试 Action格式转换功能")
    print("=" * 80)
    
    def compare_dict(actual: dict, expected: dict, path: str = "") -> tuple[bool, str]:
        """
        深度比较两个字典，返回是否相等和差异信息
        """
        if type(actual) != type(expected):
            return False, f"{path}: 类型不匹配 - 实际: {type(actual)}, 期望: {type(expected)}"
        
        if isinstance(expected, dict):
            for key in expected:
                if key not in actual:
                    return False, f"{path}.{key}: 缺失字段"
                
                is_equal, msg = compare_dict(actual[key], expected[key], f"{path}.{key}")
                if not is_equal:
                    return False, msg
            
            # 检查是否有多余的字段（除了系统字段）
            for key in actual:
                if key not in expected and not key.startswith("__"):
                    return False, f"{path}.{key}: 多余字段"
                    
        elif isinstance(expected, list):
            if len(actual) != len(expected):
                return False, f"{path}: 列表长度不匹配 - 实际: {len(actual)}, 期望: {len(expected)}"
            
            for i, (a, e) in enumerate(zip(actual, expected)):
                is_equal, msg = compare_dict(a, e, f"{path}[{i}]")
                if not is_equal:
                    return False, msg
        else:
            # 对于字符串，去除首尾空白后比较
            if isinstance(actual, str) and isinstance(expected, str):
                actual_clean = actual.strip()
                expected_clean = expected.strip()
                if actual_clean != expected_clean:
                    return False, f"{path}: 值不匹配 - 实际: '{actual_clean}', 期望: '{expected_clean}'"
            elif actual != expected:
                return False, f"{path}: 值不匹配 - 实际: {actual}, 期望: {expected}"
        
        return True, ""
    
    # 测试数据
    test_cases = [
        {
            "name": "测试1: reply格式 XML转JSON",
            "xml": """<thinking>
用户询问如何使用我们的产品，需要提供标准的产品介绍信息
</thinking>
<action name="reply">
    <goal>
    向用户介绍产品的基本功能和使用方法
    </goal>
    <types>
    standard, solution
    </types>
    <content>
    我们的产品具有强大的数据分析功能，您可以通过简单的拖拽操作来创建图表。首先，请登录您的账户，然后选择"新建项目"开始使用。
    </content>
</action>""",
            "json": {
                "thinking": "用户询问如何使用我们的产品，需要提供标准的产品介绍信息",
                "action_details": {
                    "action": "reply",
                    "goal": "向用户介绍产品的基本功能和使用方法",
                    "types": "standard, solution",
                    "content": "我们的产品具有强大的数据分析功能，您可以通过简单的拖拽操作来创建图表。首先，请登录您的账户，然后选择\"新建项目\"开始使用。"
                }
            }
        },
        {
            "name": "测试2: call_tool格式 XML转JSON",
            "xml": """<thinking>
用户需要查询特定的订单信息，我需要调用订单查询工具来获取详细数据
</thinking>
<action name="call_tool">
<goal>
通过订单ID查询用户的订单详情，为客户提供准确的订单状态信息
</goal>
<toolcalls>
<invoke name="query_order">
<id>tool_001</id>
<parameter name="order_id">ORDER123456</parameter>
<parameter name="include_history">true</parameter>
</invoke>
</toolcalls>
</action>""",
            "json": {
                "thinking": "用户需要查询特定的订单信息，我需要调用订单查询工具来获取详细数据",
                "action_details": {
                    "action": "call_tool",
                    "goal": "通过订单ID查询用户的订单详情，为客户提供准确的订单状态信息",
                    "calls": [
                        {
                            "tool_name": "query_order",
                            "tool_id": "tool_001",
                            "parameters": {
                                "order_id": "ORDER123456",
                                "include_history": "true"
                            }
                        }
                    ]
                }
            }
        },
        {
            "name": "测试2.2: call_tool格式 XML转JSON (new)",
            "xml": """\
<action name = "call_tool">  
<toolcalls>  
<invoke name="agent工具_邮箱账号无法发信原因诊断">  
<id>agent#1533</id>  
<parameter name="email">332878380@888195.top</parameter>  
</invoke>  
</toolcalls>  
</action>
""",
            "json": {
                "action_details": {
                    "action": "call_tool",
                    "calls": [
                        {
                            "tool_name": "agent工具_邮箱账号无法发信原因诊断",
                            "tool_id": "agent#1533",
                            "parameters": {
                                "email": "332878380@888195.top"
                            }
                        }
                    ]
                }
            }
        },
        {
            "name": "测试3: ask_for_help格式 XML转JSON",
            "xml": """<thinking>
用户的问题涉及复杂的技术细节，超出了我的专业范围，需要转交给技术专家处理
</thinking>
<action name="ask_for_help">
    <content>
    用户询问关于数据库索引优化的高级技术问题，涉及分布式系统架构，建议转交给技术专家团队处理
    </content>
</action>""",
            "json": {
                "thinking": "用户的问题涉及复杂的技术细节，超出了我的专业范围，需要转交给技术专家处理",
                "action_details": {
                    "action": "ask_for_help",
                    "content": "用户询问关于数据库索引优化的高级技术问题，涉及分布式系统架构，建议转交给技术专家团队处理"
                }
            }
        }
    ]
    
    # 执行测试
    total_tests = 0
    passed_tests = 0
    
    for test_case in test_cases:
        print(f"\n{'-' * 60}")
        print(f"执行 {test_case['name']}")
        print(f"{'-' * 60}")
        
        # 测试XML转JSON
        print("\n1. XML转JSON测试:")
        try:
            result = action_xml_to_json_dict(test_case["xml"])
            print(f"转换结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            # 验证核心字段
            if "error" not in result:
                print("✅ XML转JSON成功")
                passed_tests += 1
            else:
                print(f"❌ XML转JSON失败: {result['error']}")
            total_tests += 1
            
        except Exception as e:
            print(f"❌ XML转JSON异常: {str(e)}")
            total_tests += 1
        
        # 测试JSON转XML
        print("\n2. JSON转XML测试:")
        try:
            thinking = test_case["json"]["thinking"]
            action_details = test_case["json"]["action_details"]
            xml_result = action_json_dict_to_xml(thinking, action_details)
            print(f"转换结果:\n{xml_result}")
            
            # 验证生成的XML是否有效
            validation_result = action_xml_to_json_dict(xml_result)
            if "error" not in validation_result:
                print("✅ JSON转XML成功，并且生成的XML可以正确解析")
                passed_tests += 1
            else:
                print(f"❌ JSON转XML生成的XML无法解析: {validation_result['error']}")
            total_tests += 1
            
        except Exception as e:
            print(f"❌ JSON转XML异常: {str(e)}")
            total_tests += 1
    
    # 额外的边界测试
    print(f"\n{'-' * 60}")
    print("边界情况和错误处理测试")
    print(f"{'-' * 60}")
    
    # 测试空输入
    print("\n4. 空输入测试:")
    try:
        result = action_xml_to_json_dict("")
        if "error" in result:
            print("✅ 空输入正确处理")
            passed_tests += 1
        else:
            print("❌ 空输入未正确处理")
        total_tests += 1
    except Exception as e:
        print(f"❌ 空输入测试异常: {str(e)}")
        total_tests += 1
    
    # 测试格式错误的XML
    print("\n5. 错误XML格式测试:")
    try:
        malformed_xml = "<thinking>test</thinking><action>missing name attribute</action>"
        result = action_xml_to_json_dict(malformed_xml)
        if "error" in result:
            print("✅ 错误XML格式正确处理")
            passed_tests += 1
        else:
            print("❌ 错误XML格式未正确处理")
        total_tests += 1
    except Exception as e:
        print(f"❌ 错误XML格式测试异常: {str(e)}")
        total_tests += 1
    
    # 测试特殊字符
    print("\n6. 特殊字符测试:")
    try:
        special_xml = """<thinking>
包含特殊字符：&lt;test&gt; &amp; "quotes" 'apostrophes'
</thinking>
<action name="reply">
    <goal>测试特殊字符处理</goal>
    <types>standard</types>
    <content>包含&lt;&gt;&amp;"'等特殊字符的内容</content>
</action>"""
        result = action_xml_to_json_dict(special_xml)
        if "error" not in result:
            print("✅ 特殊字符正确处理")
            passed_tests += 1
        else:
            print(f"❌ 特殊字符处理失败: {result['error']}")
        total_tests += 1
    except Exception as e:
        print(f"❌ 特殊字符测试异常: {str(e)}")
        total_tests += 1
    
    # 测试HTML实体保护功能
    print(f"\n{'-' * 60}")
    print("HTML实体保护功能测试")
    print(f"{'-' * 60}")
    
    # 测试HTML实体保护
    print("\n7. HTML实体保护测试:")
    try:
        test_thinking = "测试HTML实体保护功能"
        test_action_details = {
            "action": "reply",
            "goal": "测试HTML实体保护",
            "types": ["test"],
            "content": "包含 &nbsp; 和 &copy; 实体的内容"
        }
        
        # 标准模式
        standard_xml = action_json_dict_to_xml(test_thinking, test_action_details)
        standard_result = action_xml_to_json_dict(standard_xml)
        
        # 保护模式
        protected_xml = enhanced_action_json_dict_to_xml(
            test_thinking, test_action_details, preserve_html_entities=True
        )
        protected_result = enhanced_action_xml_to_json_dict(
            protected_xml, preserve_html_entities=True
        )
        
        # 检查结果
        if "error" not in standard_result and "error" not in protected_result:
            original_content = test_action_details["content"]
            standard_content = standard_result["action_details"]["content"]
            protected_content = protected_result["action_details"]["content"]
            
            # 标准模式：HTML实体被转换
            standard_has_conversion = ("&nbsp;" in original_content and 
                                     "\xa0" in standard_content)
            
            # 保护模式：HTML实体保持原样
            protected_preserves = (original_content.strip() == protected_content.strip())
            
            if standard_has_conversion and protected_preserves:
                print("✅ HTML实体保护功能工作正常")
                print(f"   标准模式：&nbsp; → \\xa0 (转换)")
                print(f"   保护模式：&nbsp; → &nbsp; (保持)")
                passed_tests += 1
            else:
                print("❌ HTML实体保护功能异常")
                print(f"   原始: {repr(original_content)}")
                print(f"   标准: {repr(standard_content)}")
                print(f"   保护: {repr(protected_content)}")
        else:
            print("❌ HTML实体保护测试解析失败")
        
        total_tests += 1
        
    except Exception as e:
        print(f"❌ HTML实体保护测试异常: {str(e)}")
        total_tests += 1
    
    # 输出测试总结
    print(f"\n{'=' * 80}")
    print(f"测试完成")
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"成功率: {(passed_tests / total_tests * 100):.1f}%")
    print(f"{'=' * 80}")
    
    # 功能总结
    print(f"\n💡 功能说明:")
    print(f"1. 标准模式：action_json_dict_to_xml() / action_xml_to_json_dict()")
    print(f"   - 符合XML/HTML规范，HTML实体会被转换为实际字符")
    print(f"   - 性能优秀，适合大多数场景")
    print(f"")
    print(f"2. HTML实体保护模式：enhanced_*(..., preserve_html_entities=True)")
    print(f"   - 保持HTML实体的原始形式，避免二次转义")
    print(f"   - 适用于需要保持HTML实体格式的特殊场景")
    print(f"   - 性能略低，但提供更精确的内容控制")


    return passed_tests, total_tests


if __name__ == "__main__":
    import time
    from pathlib import Path
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
                '%(levelname)s\n%(message)s'
    )
    time_start = time.time()
    __test__()

    time_end = time.time()

    print("Done running file: {}\ntime: {}".format(
            Path(__file__).absolute(), time.time() - time_start))