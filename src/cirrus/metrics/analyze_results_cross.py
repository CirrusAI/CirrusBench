"""
跨版本结果对比分析脚本
找出 results_new 和 results_old 中共同存在的模型，
再找这些模型下文件名相同的结果文件，进行横向对比分析。
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from cirrus.utils.basic import load_json_dict, load_json_list

# ──────────────────────────────────────────────
# 常量配置
# ──────────────────────────────────────────────

NEW_PATH = Path("./outputs/results/results_new")
OLD_PATH = Path("./outputs/results/results_old")
TASK_FILE = Path("./benchmark_public/data/task/en_tasks_without_tool.jsonl")

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
    return dict(N=0, all_pass2=0, all_pass1=0, all_skiped=0,
                avg_pass_rate=0.0, all_NEI=0.0, out_tokens=[])


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
# 核心：跨版本查找公共模型和公共文件
# ──────────────────────────────────────────────

def find_common_models() -> List[str]:
    """找出 results_new 和 results_old 中都存在的模型目录名。"""
    new_models = {p.name for p in NEW_PATH.iterdir() if p.is_dir()} if NEW_PATH.exists() else set()
    old_models = {p.name for p in OLD_PATH.iterdir() if p.is_dir()} if OLD_PATH.exists() else set()
    common = sorted(new_models & old_models)
    return common


def find_common_files(model: str) -> List[str]:
    """对指定模型，找出 results_new 和 results_old 中文件名相同的结果文件。"""
    new_files = {fp.name for fp in (NEW_PATH / model).glob("*.json")}
    old_files = {fp.name for fp in (OLD_PATH / model).glob("*.json")}
    return sorted(new_files & old_files)


def show_overlap_summary(common_models: List[str]):
    """打印公共模型和公共文件的概览信息。"""
    new_only = {p.name for p in NEW_PATH.iterdir() if p.is_dir()} - set(common_models)
    old_only = {p.name for p in OLD_PATH.iterdir() if p.is_dir()} - set(common_models)

    print("=" * 70)
    print("【跨版本目录对比概览】")
    print("=" * 70)
    print(f"  results_new 独有模型: {sorted(new_only) or '（无）'}")
    print(f"  results_old 独有模型: {sorted(old_only) or '（无）'}")
    print(f"  共同模型 ({len(common_models)} 个): {common_models}")

    print()
    print(f"{'模型':<40} {'new文件数':>10} {'old文件数':>10} {'公共文件数':>10}")
    print("-" * 74)
    for model in common_models:
        n_new = len(list((NEW_PATH / model).glob("*.json")))
        n_old = len(list((OLD_PATH / model).glob("*.json")))
        n_common = len(find_common_files(model))
        print(f"  {model:<38} {n_new:>10} {n_old:>10} {n_common:>10}")


# ──────────────────────────────────────────────
# 分析函数
# ──────────────────────────────────────────────

def analyze_cross_models():
    """对所有公共模型，分别在公共文件集上统计 new 和 old 的指标并横向对比。"""
    common_models = find_common_models()
    if not common_models:
        print("未找到公共模型，退出。")
        return

    show_overlap_summary(common_models)

    print()
    print("=" * 80)
    print("【公共文件集 - new vs old 横向对比】")
    print("=" * 80)

    for model in common_models:
        common_files = find_common_files(model)
        if not common_files:
            print(f"\n{model}: 无公共文件，跳过")
            continue

        stats_new = _make_empty_stats()
        stats_old = _make_empty_stats()

        for fname in common_files:
            result_new = Result(load_json_dict(NEW_PATH / model / fname), model)
            result_old = Result(load_json_dict(OLD_PATH / model / fname), model)
            _accumulate(stats_new, result_new)
            _accumulate(stats_old, result_old)

        print(f"\n  [{model}]  公共文件数={len(common_files)}")
        _print_stats("    new", stats_new)
        _print_stats("    old", stats_old)


def analyze_cross_by_product():
    """对所有公共模型按产品线分层，比较 new 和 old。"""
    common_models = find_common_models()
    if not common_models:
        print("未找到公共模型，退出。")
        return

    categories = MAIN_PRODUCTS + ["其他"]

    print()
    print("=" * 80)
    print("【公共文件集 - 按产品线 new vs old 对比】")
    print("=" * 80)

    for model in common_models:
        common_files = find_common_files(model)
        if not common_files:
            continue

        stats_new: Dict[str, dict] = {cat: _make_empty_stats() for cat in categories}
        stats_old: Dict[str, dict] = {cat: _make_empty_stats() for cat in categories}

        for fname in common_files:
            r_new = Result(load_json_dict(NEW_PATH / model / fname), model)
            r_old = Result(load_json_dict(OLD_PATH / model / fname), model)
            key = r_new.cn_product if r_new.cn_product in MAIN_PRODUCTS else "其他"
            _accumulate(stats_new[key], r_new)
            _accumulate(stats_old[key], r_old)

        print(f"\n  [{model}]")
        for cat in categories:
            if stats_new[cat]["N"] == 0:
                continue
            print(f"    [{cat}]")
            _print_stats("      new", stats_new[cat])
            _print_stats("      old", stats_old[cat])


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # 1. 按模型汇总对比
    analyze_cross_models()

    # 2. 按产品线细分对比
    analyze_cross_by_product()
