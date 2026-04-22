import re
import json
import ast
import logging
from typing import List, Dict, Any, Tuple, Optional, Union


def escape_xml_characters(text: str) -> str:
    """
    转义XML特殊字符，确保XML解析正确性
    
    Args:
        text: 需要转义的文本
        
    Returns:
        转义后的安全XML文本
    """
    if not isinstance(text, str):
        return text
    
    # XML预定义的5个基本实体（必须转义）
    # 注意：必须先转义&，因为其他转义会产生&字符
    text = text.replace('&', '&amp;')   # 必须第一个
    text = text.replace('<', '&lt;')    # 必须转义，标签开始
    text = text.replace('>', '&gt;')    # 某些情况下必须转义
    text = text.replace('"', '&quot;')  # 属性值中必须转义
    text = text.replace("'", '&apos;')  # 属性值中必须转义
    
    # 处理控制字符（除了tab、换行、回车）
    # XML 1.0只允许这些字符：#x9 (tab) | #xA (LF) | #xD (CR) | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
    result = []
    for char in text:
        char_code = ord(char)
        
        # 允许的字符：tab(9), LF(10), CR(13), 普通字符(32-55295), 私有区域(57344-65533), 补充字符(65536-1114111)
        if (char_code == 0x09 or          # tab
            char_code == 0x0A or          # line feed
            char_code == 0x0D or          # carriage return
            (0x20 <= char_code <= 0xD7FF) or     # 普通字符
            (0xE000 <= char_code <= 0xFFFD) or   # 私有使用区域
            (0x10000 <= char_code <= 0x10FFFF)):  # 补充字符
            result.append(char)
        else:
            # 将非法控制字符替换为数字字符引用
            result.append(f'&#x{char_code:X};')
    
    return ''.join(result)


def unescape_xml_characters(text: str) -> str:
    """
    反转义XML字符，将XML实体转换回原始字符
    
    Args:
        text: 需要反转义的文本
        
    Returns:
        反转义后的文本
    """
    if not isinstance(text, str):
        return text
    
    # 反转义XML预定义实体
    # 注意：必须最后处理&amp;，因为其他实体包含&字符
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&apos;', "'")
    text = text.replace('&amp;', '&')    # 必须最后
    
    # 处理数字字符引用（十六进制和十进制）
    def replace_hex_entity(match):
        try:
            char_code = int(match.group(1), 16)
            if char_code <= 0x10FFFF:  # 有效Unicode范围
                return chr(char_code)
            else:
                logging.warning(f"无效的Unicode字符代码 0x{char_code:X}，保留原始实体")
                return match.group(0)
        except (ValueError, OverflowError):
            logging.warning(f"无法解析十六进制实体 {match.group(0)}，保留原始实体")
            return match.group(0)
    
    def replace_dec_entity(match):
        try:
            char_code = int(match.group(1))
            if char_code <= 0x10FFFF:  # 有效Unicode范围
                return chr(char_code)
            else:
                logging.warning(f"无效的Unicode字符代码 {char_code}，保留原始实体")
                return match.group(0)
        except (ValueError, OverflowError):
            logging.warning(f"无法解析十进制实体 {match.group(0)}，保留原始实体")
            return match.group(0)
    
    # 替换十六进制字符引用
    text = re.sub(r'&#x([0-9A-Fa-f]+);', replace_hex_entity, text)
    
    # 替换十进制字符引用
    text = re.sub(r'&#([0-9]+);', replace_dec_entity, text)
    
    # 处理常见HTML/XML实体
    common_entities = {
        '&nbsp;': '\u00A0',  # 不间断空格
        '&copy;': '\u00A9',  # 版权
        '&reg;': '\u00AE',   # 注册商标
        '&trade;': '\u2122', # 商标
        '&mdash;': '\u2014', # 长破折号
        '&ndash;': '\u2013', # 短破折号
        '&hellip;': '\u2026',# 省略号
    }
    
    for entity, replacement in common_entities.items():
        if entity in text:
            text = text.replace(entity, replacement)
    
    return text


def convert_param_types(
    param_value: Any, 
    param_type: str, 
    auto_types: List[str] = None
) -> Any:
    """
    根据指定类型转换参数值
    
    Args:
        param_value: 原始参数值
        param_type: 目标类型 ("int", "float", "bool", "list", "dict")
        auto_types: 允许自动转换的类型列表
        
    Returns:
        转换后的参数值
    """
    # 如果auto_types为空或None，或指定类型不在允许列表中，直接返回原值
    if not auto_types or param_type not in auto_types:
        return param_value
    
    if not isinstance(param_value, str):
        return param_value
    
    try:
        if param_type == "int":
            return int(param_value)
        elif param_type == "float":
            return float(param_value)
        elif param_type == "bool":
            if param_value.lower() in ("true", "yes", "1"):
                return True
            elif param_value.lower() in ("false", "no", "0"):
                return False
            return param_value
        elif param_type == "list":
            if param_value.strip().startswith('[') and param_value.strip().endswith(']'):
                try:
                    return json.loads(param_value)
                except json.JSONDecodeError:
                    return ast.literal_eval(param_value)
            else:
                return [param_value]  # 单个值转为数组
        elif param_type == "dict":
            if param_value.strip().startswith('{') and param_value.strip().endswith('}'):
                try:
                    return json.loads(param_value)
                except json.JSONDecodeError:
                    return ast.literal_eval(param_value)
            return param_value
    except (ValueError, SyntaxError) as e:
        logging.warning(f"参数类型转换失败: 无法将 '{param_value}' 转换为 {param_type}，错误: {e}")
    
    return param_value


def smart_convert_value(param_value: str, auto_types: List[str] = None) -> Any:
    """
    智能转换参数值，自动检测最合适的类型
    
    Args:
        param_value: 原始字符串值
        auto_types: 允许自动转换的类型列表
        
    Returns:
        转换后的值
    """
    if not auto_types or not isinstance(param_value, str):
        return param_value
    
    param_value = param_value.strip()
    
    # 按优先级尝试转换
    if "bool" in auto_types:
        if param_value.lower() in ("true", "yes", "1"):
            return True
        elif param_value.lower() in ("false", "no", "0"):
            return False
    
    if "int" in auto_types:
        try:
            if param_value.isdigit() or (param_value.startswith('-') and param_value[1:].isdigit()):
                return int(param_value)
        except ValueError:
            pass
    
    if "float" in auto_types:
        try:
            if '.' in param_value:
                return float(param_value)
        except ValueError:
            pass
    
    if "list" in auto_types:
        if param_value.startswith('[') and param_value.endswith(']'):
            try:
                return json.loads(param_value)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(param_value)
                except (ValueError, SyntaxError):
                    pass
    
    if "dict" in auto_types:
        if param_value.startswith('{') and param_value.endswith('}'):
            try:
                return json.loads(param_value)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(param_value)
                except (ValueError, SyntaxError):
                    pass
    
    return param_value


def find_xml_tags_with_attributes(text: str, allowed_tags: list = None) -> List[Tuple[str, int, int, int, int, Dict[str, str]]]:
    """
    查找文本中指定的XML标签的位置信息，包括属性
    返回: [(tag_name, start_tag_start, start_tag_end, end_tag_start, end_tag_end, attributes), ...]
    """
    if allowed_tags is None:
        # 如果没有限制，查找所有标签（带属性）
        pattern = r"<(\w+)([^>]*?)>(.*?)</\1>"
    else:
        # 只查找指定的标签
        tag_pattern = "|".join(re.escape(tag) for tag in allowed_tags)
        pattern = f"<({tag_pattern})([^>]*?)>(.*?)</\\1>"
    
    tags = []
    
    for match in re.finditer(pattern, text, re.DOTALL):
        tag_name = match.group(1)
        attributes_str = match.group(2).strip()
        content = match.group(3)
        
        # 解析属性
        attributes = {}
        if attributes_str:
            # 更鲁棒的属性解析：key="value" 或 key='value'，支持=前后有空格
            attr_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\')'
            for attr_match in re.finditer(attr_pattern, attributes_str):
                attr_name = attr_match.group(1)
                # Group 2 for double-quoted values, group 3 for single-quoted values
                attr_value = attr_match.group(2) if attr_match.group(2) is not None else attr_match.group(3)
                attributes[attr_name] = attr_value
        
        start_tag_start = match.start()
        # 重新计算开始标签的结束位置
        start_tag_pattern = f"<{tag_name}[^>]*>"
        start_tag_match = re.match(start_tag_pattern, text[start_tag_start:])
        if start_tag_match:
            start_tag_end = start_tag_start + start_tag_match.end()
        else:
            start_tag_end = start_tag_start + len(f"<{tag_name}>")
        
        end_tag_start = match.end() - len(f"</{tag_name}>")
        end_tag_end = match.end()
        
        tags.append((tag_name, start_tag_start, start_tag_end, end_tag_start, end_tag_end, attributes))
    
    return tags


def find_xml_tags(text: str, allowed_tags: list = None) -> List[Tuple[str, int, int, int, int]]:
    """
    向后兼容的标签查找函数
    """
    tags_with_attrs = find_xml_tags_with_attributes(text, allowed_tags)
    return [(name, start1, start2, end1, end2) for name, start1, start2, end1, end2, attrs in tags_with_attrs]


def manual_xml_parse_fallback(
    text: str, 
    restrict_tags: List[str] = None,
    auto_types: List[str] = None
) -> Dict[str, Any]:
    """
    手动XML解析回退机制，用于处理格式不完整的XML
    """
    result = {}
    
    # 基本的标签匹配模式
    if restrict_tags:
        tags_pattern = "|".join(re.escape(tag) for tag in restrict_tags)
    else:
        tags_pattern = r"\w+"
    
    # 多种标签匹配策略
    patterns = [
        rf'<({tags_pattern})>(.*?)</\1>',  # 标准XML
        rf'<({tags_pattern})\s[^>]*>(.*?)</\1>',  # 带属性的XML
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            for tag_name, content in matches:
                # 反转义内容
                content = unescape_xml_characters(content.strip())
                
                # 智能类型转换
                if auto_types:
                    content = smart_convert_value(content, auto_types)
                
                result[tag_name] = content
            break
    
    if not result and text.strip():
        # 如果完全无法解析，返回原文本
        return {"__unparsed__": text.strip()}
    
    return result


def parse_level(
    text: str, 
    restrict_tags: List[str] = None, 
    depth: int = 0,
    auto_types: List[str] = None
) -> Dict[str, Any]:
    """
    解析单层XML结构，支持类型标记、重复标签和智能内容归属
    """
    if depth > 10:  # 防止无限递归
        return {"__unparsed__": text.strip()}
    
    result = {}
    tag_counts = {}  # 记录标签出现次数
    last_end = 0
    unmatched_parts = []
    
    try:
        # 根据restrict_tags查找XML标签（带属性）
        if restrict_tags:
            # 限制模式：只解析在限制列表中的标签
            all_xml_tags = find_xml_tags_with_attributes(text, allowed_tags=None)
            # 只保留在限制列表中的标签
            xml_tags = [tag for tag in all_xml_tags if tag[0] in restrict_tags]
        else:
            xml_tags = find_xml_tags_with_attributes(text, allowed_tags=None)
        
        # 如果没有要解析的标签，尝试手动解析
        if not xml_tags:
            fallback_result = manual_xml_parse_fallback(text, restrict_tags, auto_types)
            if fallback_result:
                return fallback_result
            return {"__unparsed__": text.strip()} if text.strip() else {}
        
        # 按出现顺序排序
        xml_tags.sort(key=lambda x: x[1])
        
        for tag_name, start_tag_start, start_tag_end, end_tag_start, end_tag_end, attributes in xml_tags:
            # 保存标签前的文本
            before_text = text[last_end:start_tag_start]
            if before_text.strip():
                unmatched_parts.append(before_text)
            
            # 获取标签内容并反转义
            tag_content = text[start_tag_end:end_tag_start]
            
            # 检查是否有type属性
            force_list = attributes.get("type") == "list"
            
            # 递归解析标签内容
            if tag_content.strip():
                nested_result = parse_level(tag_content, restrict_tags, depth + 1, auto_types)
                
                # 处理递归解析结果
                if isinstance(nested_result, dict):
                    if len(nested_result) == 1 and "__unparsed__" in nested_result:
                        # 只有系统字段的情况，提取内容
                        content = unescape_xml_characters(nested_result["__unparsed__"])
                    else:
                        # 正常的嵌套结构，保留完整字典
                        content = nested_result
                else:
                    # 简单内容
                    content = unescape_xml_characters(tag_content.strip()) if isinstance(nested_result, str) else nested_result
            else:
                content = ""
            
            # 智能类型转换
            if isinstance(content, str) and auto_types:
                content = smart_convert_value(content, auto_types)
            
            # 保留属性信息 - 如果有属性，将内容包装为带属性的结构
            if attributes:
                if isinstance(content, dict):
                    # 如果内容本身是字典，添加属性信息
                    content = {**content, **{f"@{key}": value for key, value in attributes.items()}}
                else:
                    # 如果内容是简单值，创建包含属性的结构
                    content = {
                        "#text": content,
                        **{f"@{key}": value for key, value in attributes.items()}
                    }
            
            # 处理重复标签和类型标记
            tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1
            
            if tag_name in result:
                # 标签重复，转为列表
                if not isinstance(result[tag_name], list):
                    result[tag_name] = [result[tag_name]]
                result[tag_name].append(content)
            elif force_list:
                # 强制转为列表
                result[tag_name] = [content]
            else:
                # 首次出现
                result[tag_name] = content
            
            last_end = end_tag_end
        
        # 处理最后剩余的文本 - 智能归属
        remaining = text[last_end:]
        if remaining.strip():
            unmatched_parts.append(remaining)
        
        # 处理未匹配的内容
        if unmatched_parts:
            unmatched_text = "".join(unmatched_parts).strip()
            if result:
                # 有其他标签时，保留未匹配内容到系统字段
                # 在限制标签模式下，这些可能是被过滤的重要内容
                result["__unparsed__"] = unmatched_text
            else:
                # 没有任何标签，直接返回
                return {"__unparsed__": unmatched_text}
        
        return result
        
    except Exception as e:
        logging.warning(f"XML解析失败，尝试手动解析: {str(e)}")
        # 尝试手动解析作为回退
        fallback_result = manual_xml_parse_fallback(text, restrict_tags, auto_types)
        if fallback_result:
            return fallback_result
        return {"__unparsed__": text.strip(), "parse_error": str(e)}


def convert_to_openai_toolcall_format(
    parsed_result: Dict[str, Any],
    toolcall_tag: str = "toolcalls",
    invoke_tag: str = "invoke", 
    param_tag: str = "parameter"
) -> List[Dict[str, Any]]:
    """
    将解析的XML结果转换为OpenAI工具调用JSON格式
    支持嵌套结构中的工具调用
    
    Args:
        parsed_result: XML解析结果
        toolcall_tag: 工具调用根标签名
        invoke_tag: 调用标签名
        param_tag: 参数标签名
        
    Returns:
        OpenAI格式的工具调用列表
    """
    
    def find_toolcalls_recursively(data: Any, target_tag: str) -> Any:
        """递归搜索目标标签"""
        if isinstance(data, dict):
            # 直接查找目标标签
            if target_tag in data:
                return data[target_tag]
            # 递归搜索每个值
            for value in data.values():
                result = find_toolcalls_recursively(value, target_tag)
                if result is not None:
                    return result
        elif isinstance(data, list):
            # 在列表中递归搜索
            for item in data:
                result = find_toolcalls_recursively(item, target_tag)
                if result is not None:
                    return result
        return None
    
    toolcalls = []
    
    # 递归查找工具调用根节点
    toolcall_content = find_toolcalls_recursively(parsed_result, toolcall_tag)
    if not toolcall_content:
        return []
    
    # 查找invoke节点
    invoke_content = toolcall_content.get(invoke_tag)
    if not invoke_content:
        return []
    
    # 处理单个或多个invoke
    if not isinstance(invoke_content, list):
        invoke_content = [invoke_content]
    
    for invoke in invoke_content:
        # 如果invoke是字符串，需要重新解析其中的参数
        if isinstance(invoke, str):
            # 尝试从字符串中解析工具调用
            tool_call = parse_toolcall_from_string(invoke, param_tag)
            if tool_call:
                toolcalls.append(tool_call)
            continue
        
        if not isinstance(invoke, dict):
            continue
            
        # 提取工具名（新格式使用@name）
        tool_name = invoke.get("@name") or invoke.get("name")
        if not tool_name:
            continue
        
        # 提取工具ID（如果存在）
        tool_id = invoke.get("id") or invoke.get("@id")
        
        # 构建工具调用
        toolcall = {
            "name": tool_name,
            "arguments": {}
        }
        
        # 如果有ID，添加到工具调用中
        if tool_id:
            toolcall["id"] = tool_id
        
        # 提取参数 - 处理新的属性格式
        params = invoke.get(param_tag, [])
        if not isinstance(params, list):
            params = [params]
        
        for param in params:
            if isinstance(param, dict):
                # 新格式：属性在@name中，值在#text中
                param_name = param.get("@name") or param.get("name")
                param_value = param.get("#text") or param.get("value") or param.get("content", "")
                
                if param_name:
                    # 反转义参数值
                    if isinstance(param_value, str):
                        param_value = unescape_xml_characters(param_value)
                    toolcall["arguments"][param_name] = param_value
            elif isinstance(param, str):
                # 如果参数是字符串，尝试从中提取信息
                toolcall["arguments"][f"param_{len(toolcall['arguments'])}"] = param
        
        toolcalls.append(toolcall)
    
    return toolcalls


def parse_toolcall_from_string(invoke_str: str, param_tag: str = "parameter") -> Optional[Dict[str, Any]]:
    """
    从字符串中解析工具调用信息
    
    Args:
        invoke_str: 包含参数的字符串
        param_tag: 参数标签名
        
    Returns:
        解析出的工具调用信息，如果失败返回None
    """
    try:
        # 使用正则表达式提取参数
        param_pattern = rf'<{param_tag}\s+name="([^"]+)"[^>]*>(.*?)</{param_tag}>'
        param_matches = re.findall(param_pattern, invoke_str, re.DOTALL)
        
        if not param_matches:
            return None
        
        # 尝试从invoke字符串中提取工具名（如果有的话）
        # 这里假设工具名可能在字符串的某个位置被提及
        arguments = {}
        
        for param_name, param_value in param_matches:
            # 反转义参数值
            param_value = unescape_xml_characters(param_value.strip())
            arguments[param_name] = param_value
        
        # 如果成功提取到参数，返回一个通用的工具调用结构
        # 工具名暂时设为"unknown"，调用者可以根据上下文推断
        return {
            "name": "unknown_tool",
            "arguments": arguments
        }
        
    except Exception as e:
        logging.warning(f"从字符串解析工具调用失败: {str(e)}")
        return None


def parse_xml_str(
    content: str, 
    restrict_tags: List[str] = None,
    to_openai_json: bool = False,
    auto_types: List[str] = None,
    toolcall_tag: str = "toolcalls",
    invoke_tag: str = "invoke",
    param_tag: str = "parameter"
) -> Tuple[Dict[str, Any], str]:
    """
    增强的XML解析函数，支持多种功能
    
    Args:
        content: XML内容字符串
        restrict_tags: 限制只解析指定的标签，如果不指定则解析所有标签
        to_openai_json: 是否将工具调用格式转换为OpenAI JSON格式，默认False
        auto_types: 需要自动类型转换的类型列表，如["int", "bool", "list"]
        toolcall_tag: 工具调用根标签名（仅在to_openai_json=True时使用）
        invoke_tag: 调用标签名（仅在to_openai_json=True时使用）
        param_tag: 参数标签名（仅在to_openai_json=True时使用）
        
    Returns:
        Tuple[Dict[str, Any], str]: (解析结果, 状态消息)
        - 解析结果: 包含解析的XML结构，保留层次关系
        - 状态消息: 解析状态信息，用于日志或错误显示
    """
    
    try:
        # 输入验证
        if not content or not content.strip():
            return {}, "输入内容为空"
        
        # 开始解析
        start_time = logging.getLogger().level  # 简单的时间标记
        
        # 执行主要解析
        result = parse_level(
            content.strip(), 
            restrict_tags=restrict_tags, 
            auto_types=auto_types
        )
        
        # 构建状态消息
        if isinstance(result, dict):
            found_tags = list(result.keys())
            if "parse_error" in result:
                msg = f"XML解析部分成功，有错误: {result['parse_error']}。找到标签: {found_tags}"
                #logging.warning(msg)
            else:
                msg = f"XML解析成功。找到标签: {found_tags}"
                #logging.info(msg)
        else:
            msg = "XML解析返回非字典结果"
            #logging.warning(msg)
        
        # 如果需要转换为OpenAI工具调用格式
        if to_openai_json:
            try:
                toolcalls = convert_to_openai_toolcall_format(
                    result, 
                    toolcall_tag=toolcall_tag,
                    invoke_tag=invoke_tag,
                    param_tag=param_tag
                )
                
                if toolcalls:
                    # 成功转换为工具调用格式
                    converted_result = {
                        "toolcalls": toolcalls,
                        "original_parsed": result
                    }
                    msg += f" | 转换为OpenAI格式: {len(toolcalls)}个工具调用"
                    return converted_result, msg
                else:
                    # 没有找到工具调用，返回原始解析结果
                    msg += " | 未找到工具调用结构，返回原始解析结果"
                    return result, msg
                    
            except Exception as convert_error:
                # 转换失败，返回原始解析结果
                error_msg = f"工具调用格式转换失败: {str(convert_error)}"
                logging.warning(error_msg)
                msg += f" | {error_msg}，返回原始解析结果"
                return result, msg
        
        return result, msg
        
    except Exception as e:
        # 全局错误处理
        error_msg = f"XML解析发生错误: {str(e)}"
        logging.error(error_msg)
        
        # 尝试最后的手动解析
        try:
            fallback_result = manual_xml_parse_fallback(
                content.strip(), 
                restrict_tags=restrict_tags,
                auto_types=auto_types
            )
            if fallback_result:
                msg = f"{error_msg} | 手动解析回退成功"
                #logging.info(msg)
                return fallback_result, msg
        except Exception as fallback_error:
            logging.error(f"手动解析回退也失败: {str(fallback_error)}")
        
        # 最终错误返回
        error_result = {
            "error": str(e),
            "original_content": content[:200] + "..." if len(content) > 200 else content,
            "content_length": len(content)
        }
        
        return error_result, error_msg



def handle(param):
    content = param.get("content", "")
    restrict_tags = param.get("restrict_tags", None)
    to_openai_json = param.get("to_openai_json", False)
    auto_types = param.get("auto_types", [])
    toolcall_tag = param.get("toolcall_tag", "toolcalls")
    invoke_tag = param.get("invoke_tag", "invoke")
    param_tag = param.get("param_tag", "parameter")
    result, msg = parse_xml_str(
        content, restrict_tags, 
        to_openai_json=to_openai_json, auto_types=auto_types, toolcall_tag=toolcall_tag, invoke_tag=invoke_tag, param_tag=param_tag)
    output = {
        "result": result,
        "msg": msg
    }
    return output


def __test__():
    xml = """\
<action name = "call_tool">  
<toolcalls type="list">  
<invoke name="agent工具_邮箱账号无法发信原因诊断">  
<id>agent#1533</id>  
<parameter name="email">332878380@888195.top</parameter>  
</invoke>  
</toolcalls>  
</action>
"""
    result, msg = parse_xml_str(xml)
    print(result)
    print(msg)

if __name__ == "__main__":
    __test__()
