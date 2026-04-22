"""
评测结果对比分析脚本（仅对各模型共有的文件进行比较）
基于 analyze_results.py，增加：找出所有模型结果文件中文件名相同的部分，再统计分析。
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from cirrus.utils.basic import load_json_dict, load_json_list

# ──────────────────────────────────────────────
# 常量配置
# ──────────────────────────────────────────────

RESULT_PATH = Path("./outputs/results/results_new")
TASK_FILE = Path("./benchmark_public/data/task/en_tasks_without_tool.jsonl")

MODELS = [
    "gpt-5-mini-0807-global",
    "qwen3-235b-a22b-instruct-2507",
    "qwen3-235b-a22b-thinking-2507",
    "deepseek-v3.2",
    "qwen3-max",
    "deepseek-r1",
    "gpt-5.2-1211-global",
    "gpt-4o-0806",
    "gpt-5-0807-global",
]

MAIN_PRODUCTS = ["阿里邮箱", "备案", "域名与网站", "短信服务", "云服务器 ECS"]

# ──────────────────────────────────────────────
# Token 提取工具函数
# ──────────────────────────────────────────────

def _extract_tokens_openai_style(msg: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    usage_inner = (msg.get("usage") or {}).get("usage") or {}
    total = usage_inner.get("total_tokens")
    if not isinstance(total, int):
        return None
    return usage_inner.get("prompt_tokens"), usage_inner.get("completion_tokens"), total


def _extract_tokens_alibaba_style(msg: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    token_info = ((msg.get("usage") or {}).get("raw_response_data") or {}).get("tokenInfo") or {}
    return (
        token_info.get("inputTokenNum"),
        token_info.get("outputTokenNum"),
        token_info.get("allTokenNum"),
    )


# ──────────────────────────────────────────────
# 任务元数据工具
# ──────────────────────────────────────────────

_task_list: Optional[List[dict]] = None


def _get_task_list() -> List[dict]:
    global _task_list
    if _task_list is None:
        _task_list = load_json_list(TASK_FILE)
    return _task_list


def get_product_name(task_id: str) -> str:
    for task in _get_task_list():
        if task["id"] == task_id:
            return task.get("product_name", "")
    return ""


# ──────────────────────────────────────────────
# 结果处理核心类
# ──────────────────────────────────────────────

class Result:
    def __init__(self, data: dict, model: str, tool_bool: bool = False):
        self.data = data
        self.model = model
        self.tool_bool = tool_bool

        self.sub_task_nums: int = self._get_all_sub_task_nums()
        self.skip: int = data.get("skip", 0)
        self.cn_product: str = get_product_name(data["task"]["id"])

        self._compute_sub_task_pass2()
        self._compute_sub_task_pass1()
        self._compute_pass_rate()
        self._compute_NEI()

    def _get_all_sub_task_nums(self) -> int:
        key = "seperate_indices" if self.tool_bool else "seperate_indices_by_llm"
        return len(self.data.get("task", {}).get(key, []))

    def _compute_sub_task_pass2(self):
        sub_task_results = self.data.get("subtasks", [])
        self.sub_task_pass2 = []
        for i in range(self.sub_task_nums):
            if i < len(sub_task_results):
                self.sub_task_pass2.append(sub_task_results[i].get("score", -1))
            else:
                self.sub_task_pass2.append(-1)
        self.pass2 = int(bool(self.sub_task_pass2) and self.sub_task_pass2[-1] == 1)

    def _compute_sub_task_pass1(self):
        sub_task_results = self.data.get("subtasks", [])
        self.sub_task_pass1 = []
        for i in range(self.sub_task_nums):
            if i < len(sub_task_results):
                sims = sub_task_results[i].get("simulation", [])
                score = sims[0].get("score", -1) if sims else -1
                self.sub_task_pass1.append(score)
            else:
                self.sub_task_pass1.append(-1)
        self.pass1 = int(all(s == 1 for s in self.sub_task_pass1))

    def _compute_pass_rate(self):
        if self.sub_task_nums == 0:
            self.pass_rate = 0.0
            return
        pass_num = 0
        for s in self.sub_task_pass2:
            if s == 1:
                pass_num += 1
            else:
                break
        self.pass_rate = pass_num / self.sub_task_nums

    def _compute_NEI(self):
        success = self.sub_task_pass2.count(1)
        if success == 0:
            self.NEI = 0.0
        elif success == 1:
            self.NEI = 1.0
        else:
            self.NEI = self.skip / (success - 1)

    def get_output_tokens(self) -> List[Optional[int]]:
        out_tokens = []
        for subtask in self.data.get("subtasks", []):
            sims = subtask.get("simulation", [])
            if not sims:
                continue
            msgs = sims[-1].get("sub_task_messages", [])
            if not msgs:
                continue
            msg = msgs[-1]
            if self.model.startswith("gpt"):
                result = _extract_tokens_openai_style(msg)
            else:
                result = _extract_tokens_alibaba_style(msg)
            out_tokens.append(result[1] if result else None)
        return out_tokens


# ──────────────────────────────────────────────
# 统计聚合函数
# ──────────────────────────────────────────────

def _make_empty_stats() -> dict:
    return dict(
        N=0,
        all_pass2=0,
        all_pass1=0,
        all_skiped=0,
        avg_pass_rate=0.0,
        all_NEI=0.0,
        out_tokens=[],
    )


def _accumulate(stats: dict, result: Result):
    N = stats["N"]
    stats["all_pass2"] += result.pass2
    stats["all_pass1"] += result.pass1
    stats["all_skiped"] += result.skip
    stats["all_NEI"] += result.NEI
    stats["out_tokens"] += [t for t in result.get_output_tokens() if t is not None]
    stats["avg_pass_rate"] = (stats["avg_pass_rate"] * N + result.pass_rate) / (N + 1)
    stats["N"] += 1


def _print_stats(label: str, stats: dict):
    N = stats["N"]
    if N == 0:
        print(f"{label}: 无数据")
        return
    tokens = stats["out_tokens"]
    amtl = sum(tokens) / len(tokens) if tokens else float("nan")
    print(
        f"{label} | N={N} | pass1={stats['all_pass1']/N:.4f} | pass2={stats['all_pass2']/N:.4f}"
        f" | ALJ={stats['all_skiped']/N:.4f} | avg_pass_rate={stats['avg_pass_rate']:.4f}"
        f" | ANEI={stats['all_NEI']/N:.4f} | AMTL={amtl:.1f}"
    )


# ──────────────────────────────────────────────
# 核心新功能：找公共文件集合
# ──────────────────────────────────────────────

def get_common_filenames(models: List[str]) -> set:
    """
    返回所有模型结果目录中文件名（stem）的交集。
    只有每个模型都存在的文件才被纳入比较。
    """
    available_models = []
    file_sets = []

    for model in models:
        model_dir = RESULT_PATH / model
        if not model_dir.exists():
            print(f"  [跳过] 目录不存在: {model_dir}")
            continue
        names = {fp.name for fp in model_dir.glob("*.json")}
        if not names:
            print(f"  [跳过] 目录为空: {model}")
            continue
        available_models.append(model)
        file_sets.append(names)

    if not file_sets:
        return set(), []

    common = file_sets[0]
    for s in file_sets[1:]:
        common = common & s

    return common, available_models


def show_file_coverage(models: List[str], common_files: set, available_models: List[str]):
    """打印各模型文件数量及公共文件数量概览。"""
    print(f"\n{'='*70}")
    print("【文件覆盖情况】")
    print(f"{'='*70}")
    for model in available_models:
        total = len(list((RESULT_PATH / model).glob("*.json")))
        print(f"  {model}: 共 {total} 个文件，其中属于公共集 {sum(1 for f in (RESULT_PATH / model).glob('*.json') if f.name in common_files)} 个")
    print(f"\n  公共文件总数（所有模型均有）: {len(common_files)}")


# ──────────────────────────────────────────────
# 基于公共文件的分析函数
# ──────────────────────────────────────────────

def analyze_common_models(models: List[str] = None):
    """对所有模型在公共文件集上做全局汇总统计并横向比较。"""
    if models is None:
        models = MODELS

    print(f"\n{'='*80}")
    print("【公共文件集 - 全模型横向对比】")
    print(f"{'='*80}")

    common_files, available_models = get_common_filenames(models)
    if not common_files:
        print("未找到公共文件，无法进行比较。")
        return

    show_file_coverage(models, common_files, available_models)

    print(f"\n{'='*70}")
    print(f"【各模型在公共 {len(common_files)} 个任务上的指标】")
    print(f"{'='*70}")

    for model in available_models:
        stats = _make_empty_stats()
        for fname in common_files:
            fp = RESULT_PATH / model / fname
            result = Result(load_json_dict(fp), model)
            _accumulate(stats, result)
        _print_stats(model, stats)


def analyze_common_by_product(models: List[str] = None):
    """对公共文件集按产品线分层，横向比较各模型。"""
    if models is None:
        models = MODELS

    common_files, available_models = get_common_filenames(models)
    if not common_files:
        print("未找到公共文件，无法进行产品线分析。")
        return

    print(f"\n{'='*80}")
    print(f"【公共文件集 - 按产品线横向对比】  公共任务数={len(common_files)}")
    print(f"{'='*80}")

    categories = MAIN_PRODUCTS + ["其他"]

    # 以 (model, product) 为键预先构建 stats
    all_stats: Dict[str, Dict[str, dict]] = {
        model: {cat: _make_empty_stats() for cat in categories}
        for model in available_models
    }

    for fname in common_files:
        for model in available_models:
            fp = RESULT_PATH / model / fname
            result = Result(load_json_dict(fp), model)
            key = result.cn_product if result.cn_product in MAIN_PRODUCTS else "其他"
            _accumulate(all_stats[model][key], result)

    for cat in categories:
        print(f"\n  [{cat}]")
        for model in available_models:
            _print_stats(f"    {model}", all_stats[model][cat])


def analyze_common_by_checkpoint_count(models: List[str] = None):
    """对公共文件集按子任务数量（检查点数）分层，横向比较各模型。"""
    if models is None:
        models = MODELS

    common_files, available_models = get_common_filenames(models)
    if not common_files:
        print("未找到公共文件，无法进行检查点分析。")
        return

    print(f"\n{'='*80}")
    print(f"【公共文件集 - 按检查点数横向对比】  公共任务数={len(common_files)}")
    print(f"{'='*80}")

    buckets = list(range(1, 6))
    all_stats: Dict[str, Dict[int, dict]] = {
        model: {b: _make_empty_stats() for b in buckets}
        for model in available_models
    }

    for fname in common_files:
        for model in available_models:
            fp = RESULT_PATH / model / fname
            result = Result(load_json_dict(fp), model)
            bucket = min(result.sub_task_nums, 5)
            _accumulate(all_stats[model][bucket], result)

    for b in buckets:
        label = f"检查点={b}" if b < 5 else "检查点≥5"
        print(f"\n  [{label}]")
        for model in available_models:
            _print_stats(f"    {model}", all_stats[model][b])


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # 1. 公共文件集全模型横向对比
    analyze_common_models()

    # 2. 按产品线横向对比
    analyze_common_by_product()

    # 3. 按检查点数横向对比
    analyze_common_by_checkpoint_count()
