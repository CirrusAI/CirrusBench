#!/bin/bash
# 基础用法:
#   sh scripts/run.sh
#   sh scripts/run.sh --model-name qwen3-max --task-ids 0008 0009
#   sh scripts/run.sh --domain with_tool --model-name deepseek-chat --overwrite

set -e

# 切换到项目根目录（无论从哪里调用）
cd "$(dirname "$0")/.."

python src/cirrus/run.py \
  --domain no_tool \
  --model-name deepseek-chat \
  --task-ids 0008 \
  --num-trials 3 \
  --max-steps 20 \
  --max-errors 10 \
  --max-concurrency 3 \
  "$@"
