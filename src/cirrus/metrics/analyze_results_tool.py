"""
评测结果分析脚本（工具调用版）
用途：对 results_tool/ 和 results_withtool/ 目录下各模型的评测结果进行多维度统计分析
主要指标：
    TIA (Tool Invocation Rate)    — 至少发起一次工具调用的任务占比
    TSA (Tool Selection Accuracy) — 工具名称首次命中率（按 mock 维度）
    TEV (Tool Execution Validity) — 工具名称+参数完全匹配率（按 mock 维度）
    pass1 / pass2 / ALJ / ANEI   — 与 analyze_results.py 相同语义
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cirrus.utils.basic import load_json_dict

# ──────────────────────────────────────────────
# 常量配置
# ──────────────────────────────────────────────

RESULTS_TOOL_PATH = Path("./outputs/simulations_v1/results_withtool")
RESULTS_WITH_TOOL_PATH = Path("./outputs/simulations_v1/results_withtool")
MIN_FILE_COUNT = 300  # 有效模型的最小 JSON 文件数阈值

MODELS_ALL = [
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

# ──────────────────────────────────────────────
# 模型发现与筛选
# ──────────────────────────────────────────────

def discover_models(result_path: Path) -> List[str]:
    """扫描结果目录，返回所有非 test 子目录名称列表。"""
    names = []
    for item in result_path.glob("*"):
        if item.is_dir() and not item.name.endswith("test"):
            names.append(item.name)
    return names


def select_models_with_sufficient_data(
    result_path: Path,
    min_count: int = MIN_FILE_COUNT,
) -> List[str]:
    """筛选 JSON 文件数量超过 min_count 的模型列表。"""
    selected = []
    for name in discover_models(result_path):
        count = len(list((result_path / name).glob("*.json")))
        if count > min_count:
            selected.append(name)
    return selected


# ──────────────────────────────────────────────
# Token 提取工具函数
# ──────────────────────────────────────────────

def _extract_total_tokens_openai_style(
    msg: Dict[str, Any],
) -> Optional[Tuple[int, int, int]]:
    """从 OpenAI 风格的 usage 字段提取 (prompt, completion, total) token 数。

    期望结构：msg["usage"]["usage"] = {prompt_tokens, completion_tokens, total_tokens}
    """
    usage_inner = (msg.get("usage") or {}).get("usage") or {}
    total = usage_inner.get("total_tokens")
    if not isinstance(total, int):
        return None
    return (
        usage_inner.get("prompt_tokens"),
        usage_inner.get("completion_tokens"),
        total,
    )


def _extract_tokens_alibaba_style(
    msg: Dict[str, Any],
) -> Optional[Tuple[int, int, int]]:
    """从阿里云风格的 tokenInfo 字段提取 (input, output, all) token 数。

    期望结构：msg["usage"]["raw_response_data"]["tokenInfo"]
    """
    token_info = (
        ((msg.get("usage") or {}).get("raw_response_data") or {}).get("tokenInfo") or {}
    )
    return (
        token_info.get("inputTokenNum"),
        token_info.get("outputTokenNum"),
        token_info.get("allTokenNum"),
    )


# ──────────────────────────────────────────────
# 工具调用统计函数
# ──────────────────────────────────────────────

def count_tool_call_success(data: Dict[str, Any]) -> Tuple[int, int]:
    """基于 task.messages 统计工具调用成功次数与 mock 总数。

    成功判定：assistant.tool_calls 中每个 call.id 能在任意 role=="tool"
    的消息里找到匹配的 tool_call_id。

    Returns:
        (成功调用次数, tool_mocks 总数)
    """
    task = data.get("task") or {}
    tool_return_ids = {
        msg["tool_call_id"]
        for msg in task.get("messages") or []
        if msg.get("role") == "tool" and msg.get("tool_call_id")
    }
    success_count = 0
    for msg in task.get("messages") or []:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            call_id = call.get("id")
            if call_id and call_id in tool_return_ids:
                success_count += 1
    total_tool_mocks = len(data.get("tool_mocks") or [])
    return success_count, total_tool_mocks


def count_tool_call_success_from_trajectory(data: Dict[str, Any]) -> Tuple[int, int]:
    """基于 trajectory 字段统计工具调用成功次数与 mock 总数。

    Returns:
        (成功调用次数, tool_mocks 总数)
    """
    traj = data.get("trajectory") or []
    tool_return_ids = {
        m["tool_call_id"]
        for m in traj
        if m.get("role") == "tool" and m.get("tool_call_id")
    }
    success_count = 0
    for m in traj:
        if m.get("role") != "assistant":
            continue
        for call in m.get("tool_calls") or []:
            call_id = call.get("id")
            if call_id and call_id in tool_return_ids:
                success_count += 1
    total_tool_mocks = len(data.get("tool_mocks") or [])
    return success_count, total_tool_mocks


def check_tool_call_tool_mock(
    mock_item: Dict[str, Any],
    tool_call: Dict[str, Any],
) -> bool:
    """判断单个工具调用是否与 mock 期望完全匹配（工具名 + 参数）。

    Args:
        mock_item:  tool_mocks 列表中的一项，含 tool_name 和
                    tool_call.function.arguments（JSON 字符串）
        tool_call:  tool_calls 列表中的一项，含
                    tool_call.tool_calls[0].function.{name,arguments}

    Returns:
        True 当且仅当工具名相同且参数字典完全一致。
    """
    mock_tool_name = mock_item["tool_name"]
    call_fn = tool_call["tool_call"]["tool_calls"][0]["function"]
    if mock_tool_name != call_fn["name"]:
        return False
    call_tool_args = json.loads(call_fn["arguments"])
    mock_tool_args = json.loads(mock_item["tool_call"]["function"]["arguments"])
    return call_tool_args == mock_tool_args


# ──────────────────────────────────────────────
# Token 分析辅助函数
# ──────────────────────────────────────────────

def analyze_task_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """对单条任务数据进行综合统计分析（子任务、工具调用、Token）。

    Args:
        data: 含 subtasks, task, tool_calls, trajectory 等字段的任务字典

    Returns:
        包含各项统计指标的字典
    """
    stats: Dict[str, Any] = {
        "subtask_count": 0,
        "subtask_passed": 0,
        "subtask_failed": 0,
        "total_trials": 0,
        "tool_call_attempts": 0,
        "tool_call_errors": 0,
        "tool_call_successes": 0,
        "total_cost": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_all_tokens": 0,
        "has_tool_call_error_in_subtask": False,
        "trajectory_length": len(data.get("trajectory") or []),
    }

    subtasks = data.get("subtasks") or []
    stats["subtask_count"] = len(subtasks)

    for subtask in subtasks:
        if subtask.get("pass", False):
            stats["subtask_passed"] += 1
        else:
            stats["subtask_failed"] += 1

        simulations = subtask.get("simulation") or []
        stats["total_trials"] += len(simulations)

        for sim in simulations:
            for msg in sim.get("sub_task_messages") or []:
                cost = msg.get("cost")
                if cost is not None:
                    stats["total_cost"] += cost

                token_info = (
                    ((msg.get("usage") or {}).get("raw_response_data") or {})
                    .get("tokenInfo") or {}
                )
                if isinstance(token_info, dict):
                    stats["total_input_tokens"] += token_info.get("inputTokenNum", 0)
                    stats["total_output_tokens"] += token_info.get("outputTokenNum", 0)
                    stats["total_all_tokens"] += token_info.get("allTokenNum", 0)

                if msg.get("role") == "tool" and msg.get("content") == "工具调用错误":
                    stats["tool_call_errors"] += 1
                    stats["has_tool_call_error_in_subtask"] = True
                elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                    stats["tool_call_attempts"] += len(msg["tool_calls"])

    for msg in data.get("trajectory") or []:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            stats["tool_call_attempts"] += len(msg["tool_calls"])
        elif msg.get("role") == "tool":
            if msg.get("content") == "工具调用错误":
                stats["tool_call_errors"] += 1
            else:
                stats["tool_call_successes"] += 1

    if stats["tool_call_attempts"] > 0:
        inferred_success = stats["tool_call_attempts"] - stats["tool_call_errors"]
        if inferred_success >= 0:
            stats["tool_call_successes"] = inferred_success

    for item in data.get("tool_calls") or []:
        if (item.get("result") or {}).get("content") == "工具调用错误":
            stats["tool_call_errors"] += 1

    return stats


def calculate_tokens_per_subtask(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按子任务维度统计 OpenAI 风格的 Token 用量。

    Args:
        data: 含 subtasks 字段的任务字典

    Returns:
        每个子任务的 token 统计列表，含 subtask_id、总量及各 trial 明细
    """
    report = []
    for subtask in data.get("subtasks") or []:
        stats: Dict[str, Any] = {
            "subtask_id": subtask.get("subtask_id"),
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "trial_count": len(subtask.get("simulation") or []),
            "trials_detail": [],
        }
        for sim in subtask.get("simulation") or []:
            trial_prompt = trial_completion = trial_total = 0
            for msg in sim.get("sub_task_messages") or []:
                usage_data = msg.get("usage")
                if usage_data and "usage" in usage_data:
                    u = usage_data["usage"]
                    trial_prompt += u.get("prompt_tokens", 0)
                    trial_completion += u.get("completion_tokens", 0)
                    trial_total += u.get("total_tokens", 0)
            stats["total_prompt_tokens"] += trial_prompt
            stats["total_completion_tokens"] += trial_completion
            stats["total_tokens"] += trial_total
            stats["trials_detail"].append(
                {"trial": sim.get("trial"), "tokens": trial_total}
            )
        report.append(stats)
    return report


# ──────────────────────────────────────────────
# 结果处理核心类
# ──────────────────────────────────────────────

class Result:
    """封装单条任务结果，计算各项指标。

    支持 tool_bool 参数切换子任务索引字段：
        tool_bool=True  → 使用 seperate_indices（按工具分段）
        tool_bool=False → 使用 seperate_indices_by_llm（按 LLM 分段）
    """

    def __init__(self, data: Dict[str, Any], model: str, tool_bool: bool = True):
        self.data = data
        self.model = model
        self.tool_bool = tool_bool

        self.sub_task_nums: int = self._get_all_sub_task_nums()
        self.skip: int = data.get("skip", 0)

        self._compute_sub_task_pass2()
        self._compute_sub_task_pass1()
        self._compute_pass_rate()
        self._compute_NEI()

    # ---------- 内部计算 ----------

    def _get_all_sub_task_nums(self) -> int:
        """根据 tool_bool 选取对应的索引字段，返回子任务总数。"""
        key = "seperate_indices" if self.tool_bool else "seperate_indices_by_llm"
        return len((self.data.get("task") or {}).get(key) or [])

    def _compute_sub_task_pass2(self) -> None:
        """计算最终得分（各子任务最后一次 trial 的 score）。"""
        sub_task_results = self.data.get("subtasks") or []
        self.sub_task_pass2: List[int] = []
        for i in range(self.sub_task_nums):
            if i < len(sub_task_results):
                self.sub_task_pass2.append(sub_task_results[i].get("score", -1))
            else:
                self.sub_task_pass2.append(-1)
        self.pass2: int = int(
            bool(self.sub_task_pass2) and self.sub_task_pass2[-1] == 1
        )

    def _compute_sub_task_pass1(self) -> None:
        """计算首次 trial 得分（各子任务第一次 trial 的 score）。"""
        sub_task_results = self.data.get("subtasks") or []
        self.sub_task_pass1: List[int] = []
        for i in range(self.sub_task_nums):
            if i < len(sub_task_results):
                sims = sub_task_results[i].get("simulation") or []
                score = sims[0].get("score", -1) if sims else -1
                self.sub_task_pass1.append(score)
            else:
                self.sub_task_pass1.append(-1)
        self.pass1: int = int(all(s == 1 for s in self.sub_task_pass1))

    def _compute_pass_rate(self) -> None:
        """连续通过率：从第一个子任务开始累计，遇失败即停止。"""
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

    def _compute_NEI(self) -> None:
        """Normalized Efficiency Index。"""
        success = self.sub_task_pass2.count(1)
        if success == 0:
            self.NEI = 0.0
        elif success == 1:
            self.NEI = 1.0
        else:
            self.NEI = self.skip / (success - 1)

    # ---------- Token 统计 ----------

    def get_output_tokens(self) -> List[Optional[int]]:
        """返回各子任务最后一次 trial 末尾消息的 output token 数列表。"""
        out_tokens: List[Optional[int]] = []
        for subtask in self.data.get("subtasks") or []:
            sims = subtask.get("simulation") or []
            if not sims:
                continue
            msgs = sims[-1].get("sub_task_messages") or []
            if not msgs:
                continue
            msg = msgs[-1]
            if self.model.startswith("gpt"):
                result = _extract_total_tokens_openai_style(msg)
            else:
                result = _extract_tokens_alibaba_style(msg)
            out_tokens.append(result[1] if result else None)
        return out_tokens


# ──────────────────────────────────────────────
# 统计聚合辅助函数
# ──────────────────────────────────────────────

def _make_empty_stats() -> Dict[str, Any]:
    return dict(
        N=0,
        all_pass2=0,
        all_pass1=0,
        all_skiped=0,
        avg_pass_rate=0.0,
        all_NEI=0.0,
        out_tokens=[],
    )


def _accumulate(stats: Dict[str, Any], result: Result) -> None:
    N = stats["N"]
    stats["all_pass2"] += result.pass2
    stats["all_pass1"] += result.pass1
    stats["all_skiped"] += result.skip
    stats["all_NEI"] += result.NEI
    stats["out_tokens"] += [t for t in result.get_output_tokens() if t is not None]
    stats["avg_pass_rate"] = (stats["avg_pass_rate"] * N + result.pass_rate) / (N + 1)
    stats["N"] += 1


def _print_stats(label: str, stats: Dict[str, Any]) -> None:
    N = stats["N"]
    if N == 0:
        print(f"{label}: 无数据")
        return
    tokens = stats["out_tokens"]
    amtl = sum(tokens) / len(tokens) if tokens else float("nan")
    print(
        f"{label} | N={N}"
        f" | pass1={stats['all_pass1']/N:.4f}"
        f" | pass2={stats['all_pass2']/N:.4f}"
        f" | ALJ={stats['all_skiped']/N:.4f}"
        f" | avg_pass_rate={stats['avg_pass_rate']:.4f}"
        f" | ANEI={stats['all_NEI']/N:.4f}"
        f" | AMTL={amtl:.1f}"
    )


# ──────────────────────────────────────────────
# 主分析函数
# ──────────────────────────────────────────────

def analyze_tool_metrics(
    models: Optional[List[str]] = None,
    result_path: Path = RESULTS_TOOL_PATH,
) -> None:
    """遍历 results_tool/ 目录，计算并打印每个模型的 TIA/TSA/TEV 指标。

    TIA (Tool Invocation Rate):    有工具调用的任务数 / 总任务数
    TSA (Tool Selection Accuracy): 工具名称首次命中的 mock 数 / 总 mock 数
    TEV (Tool Execution Validity): 名称+参数完全匹配的 mock 数 / 总 mock 数

    Args:
        models:      待分析模型列表；为 None 时使用 MODELS_ALL
        result_path: results_tool 的根目录路径
    """
    if models is None:
        models = MODELS_ALL

    print("=" * 80)
    print("【TIA / TSA / TEV 工具调用指标】")
    print("=" * 80)

    for model in models:
        result_fd = result_path / model
        result_fps = list(result_fd.glob("*.json"))
        if not result_fps:
            print(f"{model}: 结果目录为空，跳过")
            continue

        N = len(result_fps)
        all_tool_mocks = 0
        TIA = 0        # 有工具调用的任务计数
        TSA = 0        # 工具名称命中的 mock 计数
        all_success = 0  # 完全匹配的 mock 计数（TEV 分子）

        for result_fp in result_fps:
            result_data = load_json_dict(result_fp)
            tool_mocks = result_data.get("tool_mocks") or []
            tool_calls = result_data.get("tool_calls") or []

            if len(tool_calls) != 0:
                TIA += 1

            for mock_item in tool_mocks:
                tool_name_matched = False
                for tool_call in tool_calls:
                    # TSA：对当前 mock，记录第一次工具名命中
                    if not tool_name_matched:
                        mock_name = mock_item["tool_name"]
                        call_name = tool_call["tool_call"]["tool_calls"][0]["function"]["name"]
                        if mock_name == call_name:
                            TSA += 1
                            tool_name_matched = True
                    # TEV：对当前 mock，记录第一次完全匹配（遇到即停止）
                    if check_tool_call_tool_mock(mock_item, tool_call):
                        all_success += 1
                        break

            all_tool_mocks += len(tool_mocks)

        print(f"\n{model}  (总任务数={N})")
        print(f"  TEV 分子/分母: {all_success} / {all_tool_mocks}")
        if N > 0:
            print(f"  TIA: {TIA / N:.3f}")
        if all_tool_mocks > 0:
            print(f"  TSA: {TSA / all_tool_mocks:.3f}")
            print(f"  TEV: {all_success / all_tool_mocks:.3f}")


def analyze_with_tool_results(
    models: Optional[List[str]] = None,
    result_path: Path = RESULTS_WITH_TOOL_PATH,
    tool_bool: bool = False,
) -> None:
    """遍历 results_withtool/ 目录，计算并打印 pass1/pass2/ALJ/ANEI 等指标。

    Args:
        models:      待分析模型列表；为 None 时使用 MODELS_ALL
        result_path: results_withtool 的根目录路径
        tool_bool:   传递给 Result 类；False 表示用 seperate_indices_by_llm
    """
    if models is None:
        models = MODELS_ALL

    print("=" * 80)
    print("【pass1 / pass2 / ALJ / ANEI 综合指标（results_withtool）】")
    print("=" * 80)

    for model in models:
        result_fd = result_path / model
        result_fps = list(result_fd.glob("*.json"))
        if not result_fps:
            print(f"{model}: 结果目录为空，跳过")
            continue

        stats = _make_empty_stats()
        for result_fp in result_fps:
            result_data = load_json_dict(result_fp)
            result = Result(result_data, model, tool_bool=tool_bool)
            _accumulate(stats, result)

        print(f"\n{model}  (总任务数={stats['N']})")
        _print_stats("  总览", stats)


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # 1. 筛选有足量数据的模型
    select_models = select_models_with_sufficient_data(RESULTS_TOOL_PATH)
    print(f"筛选出有效模型数量: {len(select_models)}")
    for m in select_models:
        print(f"  {m}")

    # 2. 计算 TIA / TSA / TEV（全部模型）
    analyze_tool_metrics(models=MODELS_ALL, result_path=RESULTS_TOOL_PATH)

    # 3. 计算 pass1 / pass2 / ALJ / ANEI（results_withtool）
    analyze_with_tool_results(models=MODELS_ALL, result_path=RESULTS_WITH_TOOL_PATH)
