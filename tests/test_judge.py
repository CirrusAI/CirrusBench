# -*- coding: utf-8 -*-
"""
测试 judge/scoring.py 是否能根据默认配置正常调用
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cirrus.judge.scoring import _load_judge_config, scoring_content, JUDGE_CONFIG_PATH


# ──────────────────────────────────────────────
# 1. 默认配置加载测试
# ──────────────────────────────────────────────

class TestLoadJudgeConfig:
    """测试从 judge_config.yaml 加载默认配置"""

    def test_config_file_exists(self):
        """judge_config.yaml 文件应存在"""
        assert JUDGE_CONFIG_PATH.exists(), f"配置文件不存在: {JUDGE_CONFIG_PATH}"

    def test_load_returns_dict(self):
        """_load_judge_config 应返回 dict"""
        cfg = _load_judge_config()
        assert isinstance(cfg, dict)

    def test_load_has_model(self):
        """配置中应包含 model 字段"""
        cfg = _load_judge_config()
        assert "model" in cfg, "judge_config.yaml 中缺少 model 字段"
        assert isinstance(cfg["model"], str)
        assert len(cfg["model"]) > 0

    def test_load_has_temperature(self):
        """配置中应包含 temperature 字段"""
        cfg = _load_judge_config()
        assert "temperature" in cfg, "judge_config.yaml 中缺少 temperature 字段"
        assert isinstance(cfg["temperature"], (int, float))

    def test_load_has_provider(self):
        """配置中应包含 provider 字段"""
        cfg = _load_judge_config()
        assert "provider" in cfg, "judge_config.yaml 中缺少 provider 字段"
        assert isinstance(cfg["provider"], str)
        assert len(cfg["provider"]) > 0

    def test_load_default_values(self):
        """默认配置值应符合预期"""
        cfg = _load_judge_config()
        assert cfg["provider"] == "deepseek"
        assert cfg["model"] == "deepseek-chat"
        assert cfg["temperature"] == 0.0

    def test_load_fallback_on_missing_keys(self):
        """yaml 缺少 judge 节点时应安全返回空 dict"""
        yaml_content = "other_section:\n  key: value\n"
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            cfg = _load_judge_config()
        assert cfg == {}


# ──────────────────────────────────────────────
# 2. scoring_content 使用默认配置的测试（mock LLM）
# ──────────────────────────────────────────────

MOCK_JUDGE_CFG = {"provider": "deepseek", "model": "deepseek-chat", "temperature": 0.0}

def _make_mock_llm_response(content: str):
    mock_resp = MagicMock()
    mock_resp.content = content
    return mock_resp


class TestScoringContentWithDefaultConfig:
    """测试 scoring_content 在默认配置下的行为（不发真实请求）"""

    @patch("cirrus.judge.scoring._load_judge_config", return_value=MOCK_JUDGE_CFG)
    @patch("cirrus.judge.scoring.get_judge_prompt", return_value="You are a judge.")
    @patch("cirrus.judge.scoring.call_llm")
    def test_returns_1_when_included(self, mock_call_llm, mock_prompt, mock_cfg):
        """LLM 返回含 'Included' 时应得分 1"""
        mock_call_llm.return_value = _make_mock_llm_response("Included")
        result = scoring_content("A", "B", "")
        assert result == 1

    @patch("cirrus.judge.scoring._load_judge_config", return_value=MOCK_JUDGE_CFG)
    @patch("cirrus.judge.scoring.get_judge_prompt", return_value="You are a judge.")
    @patch("cirrus.judge.scoring.call_llm")
    def test_returns_0_when_not_included(self, mock_call_llm, mock_prompt, mock_cfg):
        """LLM 返回含 'Not Included' 时应得分 0"""
        mock_call_llm.return_value = _make_mock_llm_response("Not Included")
        result = scoring_content("A", "B", "")
        assert result == 0

    @patch("cirrus.judge.scoring._load_judge_config", return_value=MOCK_JUDGE_CFG)
    @patch("cirrus.judge.scoring.get_judge_prompt", return_value="You are a judge.")
    @patch("cirrus.judge.scoring.call_llm")
    def test_returns_minus1_on_unknown_response(self, mock_call_llm, mock_prompt, mock_cfg):
        """LLM 返回无法识别的内容时应得分 -1"""
        mock_call_llm.return_value = _make_mock_llm_response("I don't know.")
        result = scoring_content("A", "B", "")
        assert result == -1

    @patch("cirrus.judge.scoring._load_judge_config", return_value=MOCK_JUDGE_CFG)
    @patch("cirrus.judge.scoring.get_judge_prompt", return_value="You are a judge.")
    @patch("cirrus.judge.scoring.call_llm")
    def test_returns_minus1_on_exception(self, mock_call_llm, mock_prompt, mock_cfg):
        """LLM 调用抛出异常时应安全返回 -1"""
        mock_call_llm.side_effect = Exception("API error")
        result = scoring_content("A", "B", "")
        assert result == -1

    @patch("cirrus.judge.scoring._load_judge_config", return_value=MOCK_JUDGE_CFG)
    @patch("cirrus.judge.scoring.get_judge_prompt", return_value="You are a judge.")
    @patch("cirrus.judge.scoring.call_llm")
    def test_uses_default_model_from_config(self, mock_call_llm, mock_prompt, mock_cfg):
        """未传 model 参数时，应使用配置文件中的默认模型"""
        mock_call_llm.return_value = _make_mock_llm_response("Included")
        scoring_content("A", "B", "")
        call_kwargs = mock_call_llm.call_args
        assert call_kwargs.kwargs.get("model") == "deepseek-chat"

    @patch("cirrus.judge.scoring._load_judge_config", return_value=MOCK_JUDGE_CFG)
    @patch("cirrus.judge.scoring.get_judge_prompt", return_value="You are a judge.")
    @patch("cirrus.judge.scoring.call_llm")
    def test_uses_provider_from_config(self, mock_call_llm, mock_prompt, mock_cfg):
        """未传 provider 时，应使用配置文件中的 provider"""
        mock_call_llm.return_value = _make_mock_llm_response("Included")
        scoring_content("A", "B", "")
        call_kwargs = mock_call_llm.call_args
        assert call_kwargs.kwargs.get("provider") == "deepseek"

    @patch("cirrus.judge.scoring._load_judge_config", return_value=MOCK_JUDGE_CFG)
    @patch("cirrus.judge.scoring.get_judge_prompt", return_value="You are a judge.")
    @patch("cirrus.judge.scoring.call_llm")
    def test_uses_default_temperature_from_config(self, mock_call_llm, mock_prompt, mock_cfg):
        """未传 llm_args 时，temperature 应来自配置文件"""
        mock_call_llm.return_value = _make_mock_llm_response("Included")
        scoring_content("A", "B", "")
        call_kwargs = mock_call_llm.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.0

    @patch("cirrus.judge.scoring._load_judge_config", return_value=MOCK_JUDGE_CFG)
    @patch("cirrus.judge.scoring.get_judge_prompt", return_value="You are a judge.")
    @patch("cirrus.judge.scoring.call_llm")
    def test_override_model(self, mock_call_llm, mock_prompt, mock_cfg):
        """显式传入 model 时应覆盖默认值"""
        mock_call_llm.return_value = _make_mock_llm_response("Included")
        scoring_content("A", "B", "", model="qwen3-max")
        call_kwargs = mock_call_llm.call_args
        assert call_kwargs.kwargs.get("model") == "qwen3-max"

    @patch("cirrus.judge.scoring._load_judge_config", return_value=MOCK_JUDGE_CFG)
    @patch("cirrus.judge.scoring.get_judge_prompt", return_value="You are a judge.")
    @patch("cirrus.judge.scoring.call_llm")
    def test_override_llm_args(self, mock_call_llm, mock_prompt, mock_cfg):
        """显式传入 llm_args 时应覆盖默认值"""
        mock_call_llm.return_value = _make_mock_llm_response("Included")
        scoring_content("A", "B", "", llm_args={"temperature": 0.7})
        call_kwargs = mock_call_llm.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
