"""
评测结果分析脚本
用途：对 results_notool/ 目录下各模型的评测结果进行多维度统计分析
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

# RESULT_PATH = Path("./outputs/simulations_v1/results_notool")
# TASK_FILE = Path("./benchmark_public/data/task/en_tasks_without_tool.jsonl")


RESULT_PATH = Path("./outputs/results/results_old")
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
    """从 OpenAI 风格的 usage 字段中提取 (in, out, total) token 数。"""
    usage_inner = (msg.get("usage") or {}).get("usage") or {}
    total = usage_inner.get("total_tokens")
    if not isinstance(total, int):
        return None
    return usage_inner.get("prompt_tokens"), usage_inner.get("completion_tokens"), total


def _extract_tokens_alibaba_style(msg: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    """从阿里云风格的 tokenInfo 字段中提取 (in, out, total) token 数。"""
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
    """根据 task_id 查找对应的产品名称，找不到返回空字符串。"""
    for task in _get_task_list():
        if task["id"] == task_id:
            return task.get("product_name", "")
    return ""


# ──────────────────────────────────────────────
# 结果处理核心类
# ──────────────────────────────────────────────

class Result:
    """封装单条任务结果，计算各项指标。"""

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

    # ---------- 内部计算 ----------

    def _get_all_sub_task_nums(self) -> int:
        key = "seperate_indices" if self.tool_bool else "seperate_indices_by_llm"
        return len(self.data.get("task", {}).get(key, []))

    def _compute_sub_task_pass2(self):
        """最终得分（最后一次 trial 的 score）。"""
        sub_task_results = self.data.get("subtasks", [])
        self.sub_task_pass2 = []
        for i in range(self.sub_task_nums):
            if i < len(sub_task_results):
                self.sub_task_pass2.append(sub_task_results[i].get("score", -1))
            else:
                self.sub_task_pass2.append(-1)

        self.pass2 = int(bool(self.sub_task_pass2) and self.sub_task_pass2[-1] == 1)

    def _compute_sub_task_pass1(self):
        """首次 trial 得分。"""
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
        """连续通过率（从第一个子任务开始，遇到失败即停止）。"""
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
        """Normalized Efficiency Index。"""
        success = self.sub_task_pass2.count(1)
        if success == 0:
            self.NEI = 0.0
        elif success == 1:
            self.NEI = 1.0
        else:
            self.NEI = self.skip / (success - 1)

    # ---------- token 统计 ----------

    def get_output_tokens(self) -> List[Optional[int]]:
        """返回每个子任务最后一次 trial 末尾消息的 output token 数列表。"""
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
    # 增量更新滚动均值，避免先累计再除
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
# 各维度分析入口
# ──────────────────────────────────────────────

def analyze_all_models():
    """对所有模型做全局汇总统计。"""
    print("=" * 80)
    print("【全模型汇总】")
    print("=" * 80)
    for model in MODELS:
        result_fps = list((RESULT_PATH / model).glob("*.json"))
        if not result_fps:
            print(f"{model}: 结果目录为空，跳过")
            continue

        stats = _make_empty_stats()
        for fp in result_fps:
            result = Result(load_json_dict(fp), model)
            _accumulate(stats, result)

        print(f"\n{model}  (总任务数={stats['N']})")
        _print_stats("  总览", stats)


def analyze_by_checkpoint_count(model: str):
    """按子任务数量（检查点数）分层统计单个模型。"""
    result_fps = list((RESULT_PATH / model).glob("*.json"))
    print(f"\n{'='*60}")
    print(f"【按检查点数分析】 模型={model}  总任务数={len(result_fps)}")
    print(f"{'='*60}")

    bucket_stats: Dict[int, dict] = {i: _make_empty_stats() for i in range(1, 6)}

    for fp in result_fps:
        result = Result(load_json_dict(fp), model)
        bucket = min(result.sub_task_nums, 5)
        _accumulate(bucket_stats[bucket], result)

    for i in range(1, 6):
        label = f"检查点={i}" if i < 5 else "检查点≥5"
        _print_stats(label, bucket_stats[i])


def analyze_by_product(model: str):
    """按产品线分层统计单个模型。"""
    result_fps = list((RESULT_PATH / model).glob("*.json"))
    print(f"\n{'='*60}")
    print(f"【按产品线分析】 模型={model}  总任务数={len(result_fps)}")
    print(f"{'='*60}")

    categories = MAIN_PRODUCTS + ["其他"]
    bucket_stats: Dict[str, dict] = {p: _make_empty_stats() for p in categories}

    for fp in result_fps:
        result = Result(load_json_dict(fp), model)
        product = result.cn_product
        key = product if product in MAIN_PRODUCTS else "其他"
        _accumulate(bucket_stats[key], result)

    for cat in categories:
        _print_stats(cat, bucket_stats[cat])


def show_product_distribution(model: str):
    """打印该模型结果集中各产品的任务数量分布。"""
    result_fps = list((RESULT_PATH / model).glob("*.json"))
    products = []
    for fp in result_fps:
        data = load_json_dict(fp)
        products.append(get_product_name(data["task"]["id"]))
    counter = Counter(products)
    print(f"\n产品分布（模型={model}）：")
    for name, cnt in counter.most_common():
        print(f"  {name}: {cnt}")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # 1. 所有模型全局汇总
    analyze_all_models()

    # 2. 对单个模型做细粒度分析（可按需修改目标模型）
    target_model = "gpt-5-0807-global"
    show_product_distribution(target_model)
    analyze_by_checkpoint_count(target_model)
    analyze_by_product(target_model)
