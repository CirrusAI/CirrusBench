#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量读取某个文件夹下的所有 JSON 文件，
只统计 subtasks -> simulation(trial) 中的 token，
不统计 trajectory 中的 token。

统计字段位置:
    subtasks[*].simulation[*].sub_task_messages[*].usage.usage
中的:
    - prompt_tokens
    - completion_tokens
    - total_tokens

用法:
    python sum_trial_tokens.py /path/to/json_folder
    python sum_trial_tokens.py /path/to/json_folder --dedup
    python sum_trial_tokens.py /path/to/json_folder --pattern "*.json"
"""

import os
import json
import argparse
from pathlib import Path


def empty_stats():
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }


def add_stats(target, src):
    target["prompt_tokens"] += src.get("prompt_tokens", 0)
    target["completion_tokens"] += src.get("completion_tokens", 0)
    target["total_tokens"] += src.get("total_tokens", 0)


def extract_trial_tokens(data, dedup=False):
    """
    只统计:
        subtasks[*].simulation[*].sub_task_messages[*].usage.usage
    中的 token 数据

    dedup=True 时，按 usage.id 去重
    """
    stats = empty_stats()
    seen_ids = set()

    subtasks = data.get("subtasks", [])
    if not isinstance(subtasks, list):
        return stats

    for subtask in subtasks:
        simulations = subtask.get("simulation", [])
        if not isinstance(simulations, list):
            continue

        for sim in simulations:
            messages = sim.get("sub_task_messages", [])
            if not isinstance(messages, list):
                continue

            for msg in messages:
                usage_wrapper = msg.get("usage")
                if not isinstance(usage_wrapper, dict):
                    continue

                # 外层 usage 里通常有 id 和内层 usage
                usage_id = usage_wrapper.get("id")
                inner_usage = usage_wrapper.get("usage")

                if not isinstance(inner_usage, dict):
                    continue

                prompt_tokens = inner_usage.get("prompt_tokens", 0) or 0
                completion_tokens = inner_usage.get("completion_tokens", 0) or 0
                total_tokens = inner_usage.get("total_tokens", 0) or 0

                if dedup and usage_id:
                    if usage_id in seen_ids:
                        continue
                    seen_ids.add(usage_id)

                stats["prompt_tokens"] += prompt_tokens
                stats["completion_tokens"] += completion_tokens
                stats["total_tokens"] += total_tokens

    return stats


def process_json_file(file_path, dedup=False):
    """
    读取单个 JSON 文件并返回 trial token 统计
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return extract_trial_tokens(data, dedup=dedup)
    except Exception as e:
        print(f"[ERROR] 读取失败: {file_path} -> {e}")
        return empty_stats()


def scan_folder(folder, pattern="*.json"):
    """
    递归扫描文件夹下所有匹配文件
    """
    folder_path = Path(folder)
    return list(folder_path.rglob(pattern))


def main():
    parser = argparse.ArgumentParser(
        description="批量统计文件夹下所有 JSON 文件中 simulation trial 的输入/输出 token（不统计 trajectory）"
    )
    parser.add_argument("folder", help="JSON 文件所在文件夹")
    parser.add_argument("--dedup", action="store_true", help="按请求 id 去重统计")
    parser.add_argument("--pattern", default="*.json", help='文件匹配模式，默认 "*.json"')
    args = parser.parse_args()

    folder = args.folder
    dedup = args.dedup
    pattern = args.pattern

    if not os.path.isdir(folder):
        print(f"[ERROR] 文件夹不存在: {folder}")
        return

    json_files = scan_folder(folder, pattern)

    if not json_files:
        print(f"[INFO] 在文件夹 {folder} 下没有找到匹配 {pattern} 的文件")
        return

    grand_stats = empty_stats()

    print("=" * 100)
    print(f"扫描目录: {folder}")
    print(f"匹配模式: {pattern}")
    print(f"统计范围: 仅 subtasks -> simulation(trial) -> sub_task_messages")
    print(f"统计模式: {'按 id 去重' if dedup else '不去重'}")
    print("=" * 100)

    for file_path in json_files:
        file_stats = process_json_file(file_path, dedup=dedup)
        add_stats(grand_stats, file_stats)

        print(
            f"{file_path}\n"
            f"  prompt_tokens     = {file_stats['prompt_tokens']}\n"
            f"  completion_tokens = {file_stats['completion_tokens']}\n"
            f"  total_tokens      = {file_stats['total_tokens']}\n"
        )

    print("=" * 100)
    print(f"文件数: {len(json_files)}")
    print(f"总输入 tokens  (prompt_tokens)     = {grand_stats['prompt_tokens']}")
    print(f"总输出 tokens  (completion_tokens) = {grand_stats['completion_tokens']}")
    print(f"总 tokens      (total_tokens)      = {grand_stats['total_tokens']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
