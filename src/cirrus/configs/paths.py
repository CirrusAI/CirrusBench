from pathlib import Path

# 当前文件: src/llm_benchmark/config/paths.py
CURRENT_FILE = Path(__file__).resolve()

# 项目根目录
ROOT_DIR = CURRENT_FILE.parents[3]


# # 常用目录
# SRC_DIR = ROOT_DIR / "src"
ALL_DATA_DIR = ROOT_DIR / "data"
CONFIGS_DIR = ROOT_DIR / "configs"


# data 子目录
# CONFIG_DIR = ROOT_DIR / "configs"
PROMPTS_DIR = ALL_DATA_DIR / "prompts"
REFERENCE_DIR = ALL_DATA_DIR / 'reference'
TASK_DIR = ALL_DATA_DIR / 'task'
# OUTPUTS_DIR = ROOT_DIR / "outputs"
# DOCS_DIR = ROOT_DIR / "docs"
# NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
# TESTS_DIR = ROOT_DIR / "tests"



# prompts 子目录
JUDGE_PROMPTS_DIR = PROMPTS_DIR / "judge"
JUDGE_PROMPT_PATH = JUDGE_PROMPTS_DIR / 'judge_prompt.md'

TASK_PROMPTS_DIR = PROMPTS_DIR / "task"
TASK_POLICY_PROMPT_PATH  = TASK_PROMPTS_DIR / 'policy.md'
