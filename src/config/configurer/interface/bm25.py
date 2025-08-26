import datetime
from abc import abstractmethod

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from . import RetrieverConfigurer
from .vector_store import VectorStoreConfigurer
from ...model.retriever.bm25 import BM25Configuration


class BM25Configurer(RetrieverConfigurer):

    @abstractmethod
    def configure(self, config: BM25Configuration, vs_configurer: VectorStoreConfigurer, /, **kwargs):
        pass

    @abstractmethod
    async def async_configure(self, config: BM25Configuration, vs_configurer: VectorStoreConfigurer, /, **kwargs):
        """
        Configures the BM25 retriever.

        This asynchronous method initializes and configures the BM25 retriever based on the provided
        `BM25Configuration`. It retrieves documents from various sources (uploaded files and
        external vector stores), chunks them, and then uses these chunks to create the BM25 retriever.

        Args:
            config: The configuration object for the BM25 retriever.
            vs_configurer: An instance of `VectorStoreConfigurer` used to retrieve
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

    @abstractmethod
    async def add_documents(self, documents: list[Document]):
        """
        Adds a list of documents to the BM25 retriever asynchronously.

        This method is expected to be implemented in a subclass, as it is abstract.
        It allows for the addition of multiple documents at once. The
        ``documents`` parameter is a list of ``Document`` objects that need to be
        processed or stored. Actual implementation details are specific to the
        concrete subclasses.

        :param documents: A list containing instances of ``Document`` that are
            appended to the existing list of documents in the BM25Retriever.
        :type documents: list[Document]
        :return: None
        """
        pass

    @abstractmethod
    async def replace_all_documents(self, documents: list[Document]):
        """
        Replaces all the current documents with the given list of `Document` objects.

        This asynchronous method ensures that all current documents of the BM25 Retriever are completely
        replaced by the provided list of new documents. All previous content is
        discarded, and only the given `Document` instances will exist in the system.

        :param documents: A list of `Document` objects to replace the current
            documents.
        :type documents: list[Document]
        :return: None
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
