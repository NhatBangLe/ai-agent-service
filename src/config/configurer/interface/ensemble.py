from abc import abstractmethod

from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import RetrieverLike
from langchain_core.tools import Tool

from . import RetrieverConfigurer


class EnsembleRetrieverConfigurer(RetrieverConfigurer):

    @abstractmethod
    def configure(self, retrievers: list[RetrieverLike], weights: list[float], **kwargs):
        pass

    async def async_configure(self, retrievers: list[RetrieverLike], weights: list[float], **kwargs):
        self.configure(**kwargs)

    @property
    @abstractmethod
    def tool(self) -> Tool | None:
        pass

    @property
    @abstractmethod
    def retriever(self) -> EnsembleRetriever | None:
        pass
