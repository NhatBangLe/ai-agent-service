import asyncio
import datetime
from abc import abstractmethod

from langchain_community.retrievers import BM25Retriever

from . import RetrieverConfigurer
from ...model.retriever.bm25 import BM25Configuration


class BM25Configurer(RetrieverConfigurer):

    def configure(self, config: BM25Configuration, /, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_configure(config, **kwargs))

    async def async_configure(self, config: BM25Configuration, /, **kwargs):
        """
        Configures the BM25 retriever.

        This asynchronous method initializes and configures the BM25 retriever based on the provided
        `BM25Configuration`. It retrieves documents from various sources (uploaded files and
        external vector stores), chunks them, and then uses these chunks to create the BM25 retriever.

        Args:
            config: The configuration object for the BM25 retriever.
            **kwargs: Additional keyword arguments.
                vs_configurer: An instance of `VectorStoreConfigurer` used to retrieve
                vector store configurations (required).
                embeddings_configurer: An instance of `EmbeddingsConfigurer` used to
                retrieve embeddings models (required).

        Raises:
            ValueError: If `vs_configurer` or `embeddings_configurer` is not provided,
                if the specified embeddings model is not configured, or if an
                unsupported document source is encountered.
        """
        pass

    @property
    @abstractmethod
    def retriever(self) -> BM25Retriever | None:
        pass

    @property
    @abstractmethod
    def last_sync(self) -> datetime.datetime | None:
        pass

    @property
    @abstractmethod
    def config(self) -> BM25Configuration | None:
        pass
