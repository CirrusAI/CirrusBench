from abc import ABC, abstractmethod
from typing import List

from cirrus.llm.schemas import Message, ModelConfig, ProviderConfig


class BaseLLMProvider(ABC):
    def __init__(self, provider_config: ProviderConfig):
        self.provider_config = provider_config

    @abstractmethod
    def generate(self, messages: List[Message], model_config: ModelConfig) -> str:
        pass
