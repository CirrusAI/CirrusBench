# -*- coding: UTF-8 -*-
"""
@Author: fdu
@CreateDate: 2023-03-29
@File: basic
@Project: StoryTelling
"""

import ast
import os
import time
import traceback
from pathlib import Path
from typing import Union, List, Callable, Mapping, Any, Dict, Tuple
import logging
import random

import numpy as np
import pandas as pd
import pickle
import json
import datetime
def show_dict_keys(
        data_dict: Dict[str, Any], indent: int = 0,
        header: str = '',
) -> str:
    """展示多层字典的keys结构，无需展示value（只需要value的具体类型"""
    string = header
    for key, value in data_dict.items():
        if isinstance(value, dict):
            string += f"{' ' * indent}{key}:\n"
            string += show_dict_keys(value, indent + 2)
        else:
            string += f"{' ' * indent}{key}: {type(value)}\n"
    # logging.info(string)
    return string

def show_dict_keys_with_json(
        data_dict: Dict[str, Any], 
        indent: int = 0,
        header: str = '',
        parse_json: bool = True,
        ) -> str:
    """展示多层字典的keys结构，无需展示value（只需要value的具体类型"""
    string = header
    if isinstance(data_dict, str):
        try:
            data_dict = parse_json_string(data_dict, show_warning=False)
            return show_dict_keys_with_json(data_dict, indent, "", parse_json)
        except Exception as e:
            return f"{data_dict[:100]}..."
            
    for key, value in data_dict.items():
        if isinstance(value, dict):
            string += f"{' ' * indent}{key}:\n"
            string += show_dict_keys_with_json(value, indent + 2, parse_json=parse_json)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            string += f"{' ' * indent}{key}: [list of dicts]\n"
            string += show_dict_keys_with_json(
                value[0], indent + 2, f"{' ' * (indent + 2)}>>>First item ({key}):\n", parse_json=parse_json)
        elif isinstance(value, list) and (len(value) and isinstance(value[0], str)):
            string += f"{' ' * indent}{key}: [list of str]\n"
            string += show_dict_keys_with_json(
                value[0], indent + 2, f"{' ' * (indent + 2)}>>>First item ({key}):\n", parse_json=True)

        elif parse_json and isinstance(value, str):
            try:
                json_value = parse_json_string(value, show_warning=False)
                string += f"{' ' * indent}{key}: [json string]\n" + \
                    show_dict_keys_with_json(
                    json_value, indent + 2, f"{' ' * (indent + 2)}>>>JSON value ({key}):\n", parse_json=True)
            except:
                string += f"{' ' * indent}{key}: {type(value)}\n"
        else:
            string += f"{' ' * indent}{key}: {type(value)}\n"
    return string


def parse_json_string(json_str: str, show_warning: bool = True) -> Any:
    """
    解析JSON字符串为Python对象
    
    Args:
        json_str: JSON格式的字符串
    
    Returns:
        Any: 解析后的对象
    
    Raises:
        ValueError: 解析失败时抛出
    """
    try:
        if not json_str or json_str == 'None':
            return None
        
        # 处理可能的转义字符
        if isinstance(json_str, str) and (json_str.startswith('"') or '\\' in json_str):
            try:
                cleaned_str = ast.literal_eval(f'"{json_str}"')\
                    .replace('\\\\n', '\\n').replace('\\\\"', '\\"')
                return json.loads(cleaned_str)
            except:
                pass
        
        return json.loads(json_str)
    except (json.JSONDecodeError, SyntaxError, ValueError) as e:
        if show_warning:
            logging.warning(f"解析JSON字符串失败: {e}, 原始字符串: {json_str[:100]}...")
        raise ValueError(f"解析JSON字符串失败: {e}")
    

def extract_json_body(json_str: str, embrace_pattern="{}") -> str:
    """
    从json字符串中提取body部分
    """
    import re
    # pattern = r"```json(.*?)```"
    match1 = re.search(r"```json(.*?)```", json_str, re.DOTALL)
    match2 = re.search(r"```JSON(.*?)```", json_str, re.DOTALL)
    match3 = re.search(r"```(.*?)```", json_str, re.DOTALL)
    if match1:
        return match1.group(1).strip()
    elif match2:
        return match2.group(1).strip()
    elif match3:
        return match3.group(1).strip()
    if embrace_pattern == "{}":
        left_brace_idx = json_str.find("{")
        right_brace_idx = json_str.rfind("}")
        if left_brace_idx != -1 and right_brace_idx != -1:
            return json_str[left_brace_idx:right_brace_idx + 1]
    elif embrace_pattern == "[]":
        left_brace_idx = json_str.find("[")
        right_brace_idx = json_str.rfind("]")
        if left_brace_idx != -1 and right_brace_idx != -1:
            return json_str[left_brace_idx:right_brace_idx + 1]
    logging.warning(f"No JSON body found in {json_str}")
    return json_str


def recursive_parse_dict(data: Union[str, dict, list]) -> dict:
    """递归解析字典，尝试将嵌套了json字符串的字典展开"""
    if hasattr(data, 'to_dict'):
        data = data.to_dict()
    if isinstance(data, str):
        try:
            data = parse_json_string(data, show_warning=False)
            return recursive_parse_dict(data)
        except Exception as e:
            return data
    elif isinstance(data, dict):
        return {k: recursive_parse_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_parse_dict(item) for item in data]
    else:
        return data


def safe_create_dataframe_from_parsed_data(
        data_list: List[Any], 
        fallback_column_name: str = 'parsed_data'
    ) -> pd.DataFrame:
    """
    从经过recursive_parse_dict解析的数据列表安全地创建DataFrame
    
    Args:
        data_list: 经过解析的数据列表
        fallback_column_name: 当数据不是字典时使用的列名
        
    Returns:
        pd.DataFrame: 创建的DataFrame
    """
    if not data_list:
        return pd.DataFrame()
    
    # 检查第一个元素的类型
    first_item = data_list[0]
    
    if isinstance(first_item, dict):
        # 检查是否所有元素都是字典
        all_dicts = all(isinstance(item, dict) for item in data_list)
        if all_dicts:
            logging.info(f"所有元素都是字典，创建多列DataFrame")
            return pd.DataFrame(data_list)
        else:
            logging.warning("数据列表中包含非字典元素，将混合类型统一处理")
            # 将非字典元素包装成字典
            normalized_data = []
            for item in data_list:
                if isinstance(item, dict):
                    normalized_data.append(item)
                else:
                    normalized_data.append({fallback_column_name: item})
            return pd.DataFrame(normalized_data)
    else:
        # 如果第一个元素不是字典，创建单列DataFrame
        logging.info(f"数据不是字典类型，创建单列DataFrame")
        return pd.DataFrame({fallback_column_name: data_list})


def expand_dict(dct: Dict):
    dct_new = {}
    for k, v in dct.items():
        if not isinstance(v, Dict):
            dct_new[k] = v
            continue
        for k1, v1 in v.items():
            kk = f'{k}.{k1}'
            dct_new[kk] = v1
    if any(isinstance(v, Dict) for v in dct_new.values()):
        return expand_dict(dct_new)
    return dct_new


def basename_clean(fp):
    """返回出去后缀的文件名"""
    return os.path.basename(fp).split(".")[0]


def save_pickle(obj, fpath: Union[str, Path]):
    """ save the object into a .pkl file
    """
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(fp: Union[str, Path]):
    with open(fp, 'rb') as f:
        res = pickle.load(f)
    return res


def reduce_lists(lists: List[List]):
    from functools import reduce
    return reduce(lambda a, b: a + b, lists)


def save_json_dict(
        dct: Union[List, Mapping], fname: Union[str, Path], encoding='utf-8',
        info=True, indent=2, **kwargs):
    with open(fname, 'w', encoding=encoding) as jsfile:
        json.dump(dct, jsfile, ensure_ascii=False, indent=indent, **kwargs)
    if info:
        logging.info(fname)


def load_json_dict(fname: Union[str, Path], encoding='utf-8'):
    with open(fname, encoding=encoding) as f:
        dct = json.load(f)
    return dct


def load_json_list(fp: Union[Path, str], encoding='utf-8') -> List[dict]:
    """Robustly load a jsonl file: skip blank lines, report parse errors.
    读取一个jsonl文件
    """
    fp = Path(fp)
    if not fp.exists():
        logging.error(f"input file does not exist: {fp}")
        return []
    size = fp.stat().st_size
    logging.info(f"input file: {fp} size={size} bytes")
    if size == 0:
        logging.warning("input file is empty")
        return []

    json_lines = []
    parse_errors = 0
    with open(fp, encoding=encoding) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                json_lines.append(obj)
            except Exception as e:
                parse_errors += 1
                if parse_errors <= 5:
                    logging.warning(f"json parse error on line {i}: {e}; line repr: {repr(line)[:200]}")
    logging.info(f"loaded {len(json_lines)} JSON objects, parse_errors={parse_errors}")
    return json_lines


def dump_json_list(lst_of_dct: List[Mapping], fp, encoding='utf-8', ensure_ascii=False, **kwargs):
    """
    Write a list of dicts to a jsonl file.
    ensure_ascii is an explicit parameter to avoid duplicating it in kwargs.
    """
    fp = Path(fp)
    write_kwargs = dict(kwargs)  # copy to avoid mutating caller's dict
    write_kwargs.pop('ensure_ascii', None)  # remove if present to avoid duplicate arg
    with open(fp, 'w', encoding=encoding) as f:
        for dct in lst_of_dct:
            f.write(json.dumps(dct, ensure_ascii=ensure_ascii, **write_kwargs) + '\n')
    logging.info(f"wrote jsonl with {len(lst_of_dct)} lines to {fp}")


def load_json_or_jsonl(fp: Union[Path, str]) -> List[dict]:
    fp = Path(fp)
    if fp.suffix == '.json':
        return load_json_dict(fp)
    elif fp.suffix == '.jsonl':
        return load_json_list(fp)
    else:
        raise ValueError(f'unsupported file type: {fp}')


def load_multiple_json_lists(fpaths: Union[List[Path], List[str]]) -> List[dict]:
    merged = []
    for fpath in fpaths:
        merged.extend(load_json_list(fpath))
    return merged


def make_nowtime_tag0(nowtime=None, brackets=False):
    if nowtime is None:
        import datetime
        nowtime = datetime.datetime.today()
    d = nowtime.strftime('%y%m%d')
    t = str(nowtime.time()).split('.')[0].replace(':', '.')
    if brackets:
        fmt = '({}-{})'
    else:
        fmt = '{}-{}'
    return fmt.format(d, t)


def make_nowtime_tag(nowtime=None, with_time=False):
    if nowtime is None:
        import datetime
        nowtime = datetime.datetime.now()
    fmt = '%y%m%d-%H.%M.%S'
    if with_time:
        return nowtime.strftime(fmt), nowtime
    return nowtime.strftime(fmt)


def handle_conflict_dirname(name: Union[str, Path]) -> Path:
    """解决文件名冲突问题，对于已经存在的文件名，在后面增加序号，返回新的文件名"""
    name_used = Path(name)
    suffix = 0
    while name_used.exists():
        suffix += 1
        name_used = Path(f'{name}{suffix}')
    return name_used


def save_multiple_sheets(
        data: Mapping[str, pd.DataFrame], fpath: Union[str, Path], **kwargs):
    """save multiple dataframes into an Excel file with multiple sheets"""
    with pd.ExcelWriter(fpath, **kwargs) as writer:
        for sheet_name, df in data.items():
            df.to_excel(writer, sheet_name=sheet_name)
    logging.info(f'saved to {fpath}\nsheets: {list(data.keys())}')


def shape_info(data):
    if hasattr(data, 'shape'):
        return data.shape
    elif isinstance(data, dict):
        return {k: shape_info(v) for k, v in data.items()}
    elif hasattr(data, '__len__'):
        return f"[{type(data[0])}]*{len(data)}" if len(data) else "[]"
    else:
        return type(data)


def try_util_success(
        call: Callable, inputs, max_n: int = 8, default=None, record_n=False,
        **kwargs
):
    """反复尝试运行某个函数 ``call(inputs)``，直到成功为止"""
    res = None
    n = 0
    while res is None:
        n += 1
        try:
            res = call(inputs, **kwargs)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
            elif n >= max_n:
                logging.warning(f'max tries reached! error: {e}')
                return (default, n) if record_n else res

    return (res, n) if record_n else res


def try_util_success_retry_if_filtered(
        call: Callable, inputs, max_n: int = 8, sleep_time=0,
        default_return=None,
        n_security_tries: int = 3,
        **kwargs):
    """反复尝试运行某个函数 ``call(inputs)``，直到成功为止"""
    res = None
    n = 0
    n_security_tries = min(n_security_tries, max_n)
    while res is None:
        n += 1
        try:
            res = call(inputs, **kwargs)
        except Exception as e:  # OpenAI 接口调用时返回内容被安全过滤，就没必要重试了
            err_info = f'Error: {e}, exception type: {type(e)}'
            if any(_ in err_info for _ in ['InvalidRequestError', 'filtered']):
                n_security_tries -= 1
                logging.error(f'[内容被过滤]: {inputs}\n{err_info}')
                logging.warning(f'尝试次数剩余: {n_security_tries}')
                if n_security_tries <= 0:
                    return default_return
            logging.warning(f'failed calling function: {call}; error: {e}; \n'
                            f'traceback: {traceback.format_exc()}; \n'
                            f'retrying...')
            if isinstance(e, KeyboardInterrupt):
                raise e
            elif n >= max_n:
                logging.warning(f'max tries reached! error: {e}')
                return res
            if sleep_time > 0:
                time.sleep(sleep_time)
    return res


def nested_dict_update(
        d: dict,
        keys: List[str],
        value: Any
):
    
    if len(keys) > 1:
        if keys[0] not in d:
            d[keys[0]] = {}
        nested_dict_update(d[keys[0]], keys[1:], value)
    else:
        try:
            # 解决json.loads()时，单引号问题，
            d[keys[0]] = ast.literal_eval(value)

            if isinstance(d[keys[0]], np.int64):
                d[keys[0]] = int(d[keys[0]])
        except Exception as e:
            # logging.warning(f"Failed to convert {value} to dict: {e}")
            if isinstance(value, np.int64):
                d[keys[0]] = int(value)
            elif isinstance(value, np.bool_):
                 d[keys[0]] = bool(value)
            else:
                d[keys[0]] = value


def read_excel_to_dicts(
        file_path_or_df: Union[Path, str, pd.DataFrame],
        sheet_name: Union[str, int, List] = 0,
        header_row: int = 0
) -> List[Dict]:
    """支持直接传入pd.DataFrame，也支持传入文件路径"""
    if isinstance(file_path_or_df, pd.DataFrame):
        df = file_path_or_df
    else:
        df = pd.read_excel(
            file_path_or_df, sheet_name=sheet_name, header=header_row)
    df = df.fillna('')
    if 'element.explanation.1' in df.columns:
        df = df.drop(columns='element.explanation.1')
    df = df.loc[:, ~df.columns.duplicated()]
    for col in df.columns:
        if '.' in col:
            nested_keys = col.split('.')
            if nested_keys[0] not in df.columns:
                df[nested_keys[0]] = [{} for _ in range(len(df))]
            # for i in range(len(df)):  # may cause index error
            for i in df.index:
                nested_dict_update(df.at[i, nested_keys[0]], nested_keys[1:],
                                   df.at[i, col])
            df = df.drop(columns=col)
        else:
            for i in df.index:
                try:
                    df.at[i, col] = ast.literal_eval(df.at[i, col])
                except:
                    continue
    result = df
    return result.to_dict(orient='records')


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert_dict_to_excel(
        data: List[Dict],
        file_path: Union[Path, str],
        sheet_name: str = 'Sheet1'
):
    # 如果是嵌套字典，展开 用.连接
    data = [flatten_dict(d) for d in data]
    df = pd.DataFrame(data)
    df.to_excel(file_path, sheet_name=sheet_name, index=False)


def describe_nowtime(
        nowtime: datetime.datetime = None,
        fmt="%Y年%m月%d日，%H点%M分",
        return_time=False
) -> Union[str, Tuple[str, datetime.datetime]]:
    weekdays = (
        "星期一",
        "星期二",
        "星期三",
        "星期四",
        "星期五",
        "星期六",
        "星期天",
    )
    if nowtime is None:
        nowtime = datetime.datetime.now()
    # age = nowtime - npc_birthday
    # age_desc = describe_age(age)
    time_desc = "当前时间：{now} ({weekday})。".format(
        now=nowtime.strftime(fmt),
        weekday=weekdays[nowtime.weekday()],
    )
    if return_time:
        return time_desc, nowtime
    return time_desc


def generate_time_str(
        num: int = 100,
        start_time: datetime.datetime = None,
        end_time: datetime.datetime = None,
        fmt="%Y年%m月%d日，%H点%M分",
        sort=True
):
    """默认时间范围：24年11月~25年8月
    在给定时间范围中随机生成时间字符串
    """
    if start_time is None:
        start_time = datetime.datetime(2024, 11, 1)
    if end_time is None:
        end_time = datetime.datetime(2025, 8, 1)
    logging.info(f"start_time: {start_time}; end_time: {end_time}")
    start_timestamp = start_time.timestamp()
    end_timestamp = end_time.timestamp()
    timestamp_list = np.random.uniform(start_timestamp, end_timestamp, num)
    if isinstance(timestamp_list, float):
        timestamp_list = [timestamp_list]
    if sort:
        timestamp_list.sort()
    time_str_list = [
        describe_nowtime(datetime.datetime.fromtimestamp(int(t)), fmt=fmt)
        for t in timestamp_list
    ]
    return time_str_list



def sample_one_and_check(
        items: List[dict], 
        title: str = "Sampled Item Check",
        word_wrap: bool = False,
) -> dict:
    """
    sample one item from items, and check if it has anything wrong
    """

    item = random.choice(items)

    try:
        from rich import print
        from rich.panel import Panel
        from rich.syntax import Syntax
        item_str = json.dumps(item, indent=2, ensure_ascii=False)
        syntax = Syntax(item_str, "json", theme="monokai", line_numbers=True, word_wrap=word_wrap)
        print(Panel(syntax, title=title, border_style="green", expand=True))
    except ImportError:
        # fallback to simple print if rich is not installed
        print("\n" + "="*25 + f" {title} " + "="*25)
        print(json.dumps(item, indent=2, ensure_ascii=False))
        print("="*70 + "\n")

    return item


def show_pretty_dict(item: dict, title: str = "Item check", indent: int = 2, word_wrap: bool = False):
    try:
        from rich import print
        from rich.panel import Panel
        from rich.syntax import Syntax
        item_str = json.dumps(item, indent=indent, ensure_ascii=False)
        syntax = Syntax(item_str, "json", theme="monokai", line_numbers=True, word_wrap=word_wrap)
        print(Panel(syntax, title=title, border_style="green", expand=True))
    except ImportError:
        # fallback to simple print if rich is not installed
        print("\n" + "="*25 + f" {title} " + "="*25)
        print(json.dumps(item, indent=indent, ensure_ascii=False))
        print("="*70 + "\n")

    return item


def multilayer_get_item(
        item: dict,
        key: Union[str, List[str]],
) -> Any:
    """
    get item from dict by key, support nested keys
    """
    if isinstance(key, str):
        keys = key.split('.')
    else:
        keys = key
    for k in keys:
        if not isinstance(item, dict):
            logging.warning(f"sub-item is not a dict: {item}")
            return None
        item = item.get(k)
    return item


def inspect_distribution(
        items: List[dict],
        label_key_or_fn: Union[str, Callable[[dict], str]],
        as_df: bool = True,
        tag: str = "",
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    inspect the distribution of a key in items
    """
    from collections import defaultdict

    _tag = ""
    if isinstance(label_key_or_fn, str):
        label_key = label_key_or_fn
        label_fn = lambda x: multilayer_get_item(x, label_key)
        _tag = f"label_key: {label_key}"
    else:
        label_fn = label_key_or_fn
        _tag = f"label_fn: {label_fn.__name__}"
    tag = tag or _tag

    label_to_count = defaultdict(int)
    for item in items:
        label = label_fn(item)
        label_to_count[label] += 1

    label_to_count = dict(label_to_count)
    label_to_freq = {k: v / len(items) for k, v in label_to_count.items()}

    if as_df:
        try: # using pandas
            from pandas import DataFrame
            df = DataFrame({
                "count": label_to_count,
                "freq": label_to_freq,
            })
            logging.info(f"inspect_distribution ({tag}):\n{df}")
            return df
        except Exception as e:
            logging.warning(f"error in as_df=True: {e}")
    logging.info(f"inspect_distribution ({tag}):\n{label_to_count}\n{label_to_freq}")
    return label_to_count, label_to_freq


def balance_by_label_probs(
        items: List[dict],
        label_fn: Callable[[dict], str],
        prob_dict: Dict[str, float],
        default_prob: float = 1.0,
) -> List[dict]:
    """
    balance the items by the label_fn and prob_dict
    """
    from collections import defaultdict
    # label_to_count = inspect_distribution(items, label_fn, as_df=False)
    # logging.info(f"label_to_count (before balance): {label_to_count}")

    groups = defaultdict(list)
    for item in items:
        label = label_fn(item)
        groups[label].append(item)
    
    balanced_items = []
    for label, group in groups.items():
        prob = max(0, prob_dict.get(label, default_prob))
        if prob > 1.0:
            for _ in range(int(prob)):
                balanced_items.extend(group.copy())
            prob = prob - int(prob)
            
        n_samples = max(1, int(prob * len(group)))
        sampled_items = random.sample(group, n_samples)
        balanced_items.extend(sampled_items)
    
    return balanced_items



