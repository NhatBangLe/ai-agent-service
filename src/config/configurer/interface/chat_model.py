from abc import abstractmethod
from typing import Sequence

from langchain_core.language_models import BaseChatModel

from src.config.configurer.interface import Configurer
from src.config.model.chat_model import ChatModelConfiguration


class ChatModelConfigurer(Configurer):

    @abstractmethod
    def configure(self, config: ChatModelConfiguration, /, **kwargs):
        """
        Configures the language model (LLM) for the agent.

        This method configures the chat model.

        Args:
            config: The LLM configuration object.

        Raises:
            NotImplementedError: If the LLM provider specified in the configuration
                                is not currently supported.
        """
        pass

    @abstractmethod
    async def async_configure(self, config: ChatModelConfiguration, /, **kwargs):
        pass

    @abstractmethod
    def get_models(self) -> Sequence[BaseChatModel]:
        pass

    @abstractmethod
    def get_model(self, unique_name: str) -> BaseChatModel | None:
        pass

    @abstractmethod
    def get_config(self, unique_name: str) -> ChatModelConfiguration | None:
        pass
