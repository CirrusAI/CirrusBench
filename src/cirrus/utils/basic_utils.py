import json
import ast
import pickle
import datetime
import pandas as pd
import numpy as np
from typing import List, Optional, Any, Dict, Tuple, Literal, Union, Mapping, Annotated, Callable
from pathlib import Path
from pydantic import BaseModel, Field
import logging
import time
import os


RoleType = Literal['tool', 'user', 'assistant']

class FunctionCall(BaseModel):
    name: str
    arguments: str 
    def __str__(self) -> str:
        return self.model_dump_json(indent=2)

class ToolCall(BaseModel):
    id : str
    type : str  = 'function'
    function : FunctionCall
    def __str__(self) -> str:
        return self.model_dump_json(indent=2)




class TimelineMessage(BaseModel):
    message_role: RoleType
    message_type:str
    mask_content:Optional[str] = None
    mask_tool_calls:Optional[Union[str,list]] = None 
    ts:Optional[str] = None
    trace_id:Optional[str] = None
    message_raw_info:Optional[str] = None

    def __str__(self) -> str:
        return self.model_dump_json(indent=2)


def print_dict(test_dict):
    print(json.dumps(test_dict,indent=4,ensure_ascii=False))



def format_enriched_dialog(messages: List[TimelineMessage]) -> str:
    """
    输出风格：每条消息为 「角色」：内容
    - 角色用中文（user -> 客户, assistant -> 客服, tool -> 工具返回）
    - 优先使用脱敏字段 mask_content / mask_tool_calls，其次使用 content / tool_calls
    - 对 tool_calls 尝试解析为 JSON 并美化输出，若解析失败则保留原样字符串
    - 消息间用一个空行分隔
    """
    role_map = {"user": "客户", "assistant": "客服", "tool": "工具返回",} #cse数据
    #role_map = {   "客户": "客户","服务支持": "客服","监督员": "监督员"}

    def _format_tool_content(tc):
        if isinstance(tc, (list, dict)):
            try:
                return json.dumps(tc, ensure_ascii=False, indent=2)
            except Exception:
                return str(tc)
        if isinstance(tc, str):
            # 尝试把字符串解析为 JSON 再美化
            try:
                parsed = json.loads(tc)
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:
                return tc
        return str(tc)

    lines = []
    if not messages:
        return ""

    for msg in messages:
        role = role_map.get(msg.message_role)

        if msg.message_role == 'assistant':
            if msg.mask_tool_calls:
                content = _format_tool_content(msg.mask_tool_calls)
            else:
                content = msg.mask_content
        #     content = str(get_tool_calls_from_message(msg))
        elif msg.message_role == 'assistant' and msg.message_type != 'reply':
            content = _format_tool_content(msg.mask_tool_calls)
        elif msg.mask_content:
            content = msg.mask_content

        else:
            # 如果没有任何内容字段，跳过该消息
            continue

        # 用书名号 + 全角冒号
        lines.append(f"「{role}」：{content}")

    return "\n\n".join(lines)


def separate_tool_messages(messages: List[TimelineMessage]) -> Tuple[List[TimelineMessage], List[TimelineMessage]]:
    """
    将消息列表分离为“工具消息”和“剩余消息”两个列表。
    Args:
        messages (List[Message]): 原始的消息对象列表。
    Returns:
        Tuple[List[Message], List[Message]]: 一个元组，包含两个列表：
                                             - 第一个列表是所有的 'tool' 消息。
                                             - 第二个列表是所有剩余的 'user' 和 'assistant' 消息。
    """
    tool_messages = []
    other_messages = []
    for message in messages:
        if message.message_role == 'tool':
            tool_messages.append(message)
        else:
            other_messages.append(message)
            
    return tool_messages, other_messages


def merge_messages_with_tool(agent_messages:List[TimelineMessage],tool_messages:List[TimelineMessage]) -> List[TimelineMessage]:
    ### 从agent_messages中添加上工具返回
    TOOL_CALL_TYPE = {'callout', 'agent_tool', 'copilot_search', 'sop_tool', 'copilot_qa'}
    new_messages = []
    for message in agent_messages:
        new_messages.append(message)
        if message.message_type in TOOL_CALL_TYPE:
            trace_id = message.trace_id
            for tool_message in tool_messages:
                if tool_message.trace_id == trace_id:
                    new_messages.append(tool_message)
                    break
    return new_messages



class OdysseyTask(BaseModel):
    """
    A task for a Odyssey domain.
    """

    id: str = Field(description="The unique identifier for the task.")
    messages: Annotated[
        Optional[List],
        Field(
            description='Containing raw messages used to create history dialog of subtask.',
            default = None,
        ),
    ]

    seperate_indices: Annotated[
        Optional[List],
        Field(
            description='Where to create subtask',
            default = [],
        ),
    ]

    seperate_indices_by_llm: Annotated[
        Optional[List],
        Field(
            description='Where to create subtask',
            default = [],
        ),
    ]


# Utility functions from basic.py
def show_dict_keys(data_dict: Dict[str, Any], indent: int = 0, header: str = '') -> str:
    """展示多层字典的keys结构，无需展示value（只需要value的具体类型"""
    string = header
    for key, value in data_dict.items():
        if isinstance(value, dict):
            string += f"{' ' * indent}{key}:\n"
            string += show_dict_keys(value, indent + 2)
        else:
            string += f"{' ' * indent}{key}: {type(value)}\n"
    return string


def parse_json_string(json_str: str, show_warning: bool = True) -> Any:
    """Parse JSON string with multiple fallback strategies"""
    if not isinstance(json_str, str):
        return json_str

    json_str = json_str.strip()
    if not json_str:
        return {}

    # Try standard JSON first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON body
    try:
        json_body = extract_json_body(json_str)
        return json.loads(json_body)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try ast.literal_eval for Python literals
    try:
        return ast.literal_eval(json_str)
    except (ValueError, SyntaxError):
        pass

    if show_warning:
        logging.warning(f"Failed to parse JSON string: {json_str[:100]}...")
    return json_str


def extract_json_body(json_str: str, embrace_pattern="{}"):
    """Extract JSON body from string using embrace pattern"""
    left_bracket, right_bracket = embrace_pattern[0], embrace_pattern[-1]

    start_idx = json_str.find(left_bracket)
    if start_idx == -1:
        raise ValueError(f"No {left_bracket} found in string")

    bracket_count = 0
    for i, char in enumerate(json_str[start_idx:], start_idx):
        if char == left_bracket:
            bracket_count += 1
        elif char == right_bracket:
            bracket_count -= 1
            if bracket_count == 0:
                return json_str[start_idx:i+1]

    raise ValueError("Unmatched brackets")


def load_json_dict(fname: Union[str, Path], encoding='utf-8') -> Dict:
    """Load JSON dictionary from file"""
    with open(fname, 'r', encoding=encoding) as f:
        return json.load(f)


def save_json_dict(data: Dict, fname: Union[str, Path], encoding='utf-8', ensure_ascii=False, **kwargs):
    """Save dictionary as JSON file"""
    with open(fname, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, **kwargs)


def load_json_list(fp: Union[Path, str], encoding='utf-8') -> List[dict]:
    """Load list of dictionaries from JSON/JSONL file"""
    fp = Path(fp)

    if fp.suffix == '.jsonl':
        # Handle JSONL format
        data = []
        with open(fp, 'r', encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse line {line_num} in {fp}: {e}")
        return data
    else:
        # Handle regular JSON format
        with open(fp, 'r', encoding=encoding) as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError(f"Unexpected data type: {type(data)}")


def dump_json_list(lst_of_dct: List[Mapping], fp, encoding='utf-8', ensure_ascii=False, **kwargs):
    """Save list of dictionaries to JSON file"""
    fp = Path(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)

    with open(fp, 'w', encoding=encoding) as f:
        json.dump(lst_of_dct, f, ensure_ascii=ensure_ascii, **kwargs)


def save_pickle(obj, fpath: Union[str, Path]):
    """Save object to pickle file"""
    fpath = Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(fp: Union[str, Path]):
    """Load object from pickle file"""
    with open(fp, 'rb') as f:
        return pickle.load(f)


def flatten_dict(d, parent_key='', sep='.'):
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def make_nowtime_tag(nowtime=None, with_time=False):
    """Generate timestamp tag"""
    if nowtime is None:
        nowtime = datetime.datetime.now()

    if with_time:
        return nowtime.strftime("%Y%m%d_%H%M%S")
    else:
        return nowtime.strftime("%Y%m%d")


def multilayer_get_item(data_dict: Dict, key_path: str, default=None, sep='.'):
    """Get value from nested dictionary using dot notation"""
    keys = key_path.split(sep)
    current = data_dict

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current



