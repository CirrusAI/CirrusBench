# -*- coding: utf-8 -*-
"""
LLM 配置类 — 从 providers_config.yaml 加载提供商配置
"""

import yaml
from cirrus.configs.paths import CONFIGS_DIR


class LLMConfig:
    """LLM 配置类，从 configs/providers_config.yaml 加载提供商信息"""

    _config_path = CONFIGS_DIR / "providers_config.yaml"

    with open(_config_path, "r", encoding="utf-8") as _f:
        PROVIDERS: dict = yaml.safe_load(_f)
    