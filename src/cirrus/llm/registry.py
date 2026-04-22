from cirrus.configs.paths import API_KEYS_YAML, MODELS_YAML
from cirrus.llm.schemas import ModelConfig, ProviderConfig
from cirrus.llm.utils import load_yaml


def get_model_config(model_name: str) -> ModelConfig:
    data = load_yaml(MODELS_YAML)
    if model_name not in data:
        raise ValueError(f"Model '{model_name}' not found in {MODELS_YAML}")

    cfg = data[model_name] or {}
    known_keys = {"provider", "model", "temperature", "max_tokens", "top_p"}
    extra = {k: v for k, v in cfg.items() if k not in known_keys}

    return ModelConfig(
        provider=cfg["provider"],
        model=cfg["model"],
        temperature=cfg.get("temperature", 0.0),
        max_tokens=cfg.get("max_tokens", 1024),
        top_p=cfg.get("top_p"),
        extra=extra,
    )


def get_provider_config(provider_name: str) -> ProviderConfig:
    data = load_yaml(API_KEYS_YAML)
    if provider_name not in data:
        raise ValueError(f"Provider '{provider_name}' not found in {API_KEYS_YAML}")

    cfg = data[provider_name] or {}
    known_keys = {"api_key", "base_url"}
    extra = {k: v for k, v in cfg.items() if k not in known_keys}

    return ProviderConfig(
        api_key=cfg.get("api_key", ""),
        base_url=cfg.get("base_url"),
        extra=extra,
    )
