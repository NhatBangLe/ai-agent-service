from abc import abstractmethod
from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from .bm25 import BM25Configurer
from .recognizer.image import ImageRecognizerConfigurer
from .vector_store import VectorStoreConfigurer
from ..interface import Configurer
from ...model.agent import AgentConfiguration
from ....process.recognizer.image import ImageRecognizer


class AgentConfigurer(Configurer):

    @abstractmethod
    def configure(self, **kwargs):
        pass

    @abstractmethod
    async def async_configure(self, **kwargs):
        pass

    @abstractmethod
    async def reload_bm25_retriever(self):
        pass

    @property
    @abstractmethod
    def tools(self) -> Sequence[BaseTool]:
        pass

    @property
    @abstractmethod
    def chat_model(self) -> BaseChatModel:
        pass

    @property
    @abstractmethod
    def config(self) -> AgentConfiguration | None:
        pass

    @property
    @abstractmethod
    def image_recognizer(self) -> ImageRecognizer | None:
        pass

    @property
    @abstractmethod
    def vector_store_configurer(self) -> VectorStoreConfigurer | None:
        pass

    @property
    @abstractmethod
    def checkpointer(self) -> AsyncPostgresSaver | None:
        pass

    @property
    @abstractmethod
    def bm25_configurer(self) -> BM25Configurer | None:
        pass

    @property
    @abstractmethod
    def image_recognizer_configurer(self) -> ImageRecognizerConfigurer | None:
        pass
