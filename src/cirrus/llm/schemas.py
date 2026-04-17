from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ModelConfig:
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    api_key: str
    base_url: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
