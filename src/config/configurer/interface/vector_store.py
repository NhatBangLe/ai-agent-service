from abc import abstractmethod
from typing import Sequence

from langchain_core.vectorstores import VectorStore

from src.config.configurer.interface import Configurer
from src.config.model.retriever.vector_store import VectorStoreConfiguration


class VectorStoreConfigurer(Configurer):

    @abstractmethod
    def configure(self, config: VectorStoreConfiguration, /, **kwargs):
        """
        Configures a vector store based on the provided configuration. This method supports
        various types of vector store configurations and manages their specific setup processes.

        Args:
            config: The configuration object for the vector store.
                    This object dictates the type of vector store to be configured
                    and includes all necessary parameters for its setup.

        Raises:
            NotImplementedError: If the provided `config` type is not supported. Currently,
                                 only `ChromaVSConfiguration` is explicitly supported.
        """
        pass

    @abstractmethod
    async def async_configure(self, config: VectorStoreConfiguration, /, **kwargs):
        """
        Async-configures a vector store based on the provided configuration. This method supports
        various types of vector store configurations and manages their specific setup processes.

        Args:
            config: The configuration object for the vector store.
                    This object dictates the type of vector store to be configured
                    and includes all necessary parameters for its setup.

        Raises:
            NotImplementedError: If the provided `config` type is not supported. Currently,
                                 only `ChromaVSConfiguration` is explicitly supported.
        """
        pass

    @abstractmethod
    def get_store(self, unique_name: str) -> VectorStore | None:
        pass

    @abstractmethod
    def get_store_config(self, unique_name: str) -> VectorStoreConfiguration | None:
        pass

    @abstractmethod
    def get_all_stores(self) -> Sequence[VectorStore]:
        pass

    @abstractmethod
    def get_all_configs(self) -> Sequence[VectorStoreConfiguration]:
        pass
