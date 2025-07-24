import asyncio
from logging import Logger, getLogger
from os import path
from typing import Sequence

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.config.configurer import Configurer
from src.config.configurer.embeddings import EmbeddingsConfigurer
from src.config.model.retriever.vector_store import VectorStoreConfiguration
from src.config.model.retriever.vector_store.chroma import ChromaVSConfiguration
from src.util.function import get_config_folder_path


class VectorStoreConfigurer(Configurer):
    _vector_stores: dict[str, tuple[VectorStoreConfiguration, VectorStore]] | None = None
    _logger: Logger = getLogger(__name__)

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
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_configure(config, **kwargs))

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
        self._logger.debug(f"Configuring vector store {config.name}...")
        if ("embeddings_configurer" not in kwargs or
                not isinstance(kwargs["embeddings_configurer"], EmbeddingsConfigurer)):
            raise ValueError(f'Configure BM25 Retriever must provide a EmbeddingsConfigurer.')
        if self._vector_stores is None:
            self._vector_stores = {}

        # Configures Embeddings model
        embeddings_configurer: EmbeddingsConfigurer = kwargs["embeddings_configurer"]
        embeddings_config = config.embeddings_model
        await embeddings_configurer.async_configure(embeddings_config)
        embeddings_model = embeddings_configurer.get_model(embeddings_config.name)
        if embeddings_model is None:
            raise ValueError(f'No {config.embeddings_model} embeddings model has configured yet.')

        # Configures Vector Store
        if isinstance(config, ChromaVSConfiguration):
            store = self._configure_chroma(config, embeddings_model)
        else:
            raise NotImplementedError(f'{type(config)} is not supported.')

        self._vector_stores[config.name] = (config, store)
        self._logger.debug(f"Configured vector store {config.name} successfully.")

    def destroy(self, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_destroy(**kwargs))

    async def async_destroy(self, **kwargs):
        pass

    def get_store(self, unique_name: str) -> VectorStore | None:
        if self._vector_stores is None:
            self._logger.debug("No stores has been configured yet.")
            return None
        value = self._vector_stores[unique_name]
        return value[1] if value is not None else None

    def get_store_config(self, unique_name: str) -> VectorStoreConfiguration | None:
        if self._vector_stores is None:
            self._logger.debug("No stores have been configured yet.")
            return None
        value = self._vector_stores[unique_name]
        return value[0] if value is not None else None

    def get_all_stores(self) -> Sequence[VectorStore]:
        if self._vector_stores is None:
            return []
        return [store for _, (_, store) in self._vector_stores.items()]

    def get_all_configs(self) -> Sequence[VectorStoreConfiguration]:
        if self._vector_stores is None:
            return []
        return [config for _, (config, _) in self._vector_stores.items()]

    def _configure_chroma(self, config: ChromaVSConfiguration, embeddings_model: Embeddings):
        persist_dir = path.join(get_config_folder_path(), "vector_store", config.name)
        settings = chromadb.Settings(anonymized_telemetry=False)

        if config.mode == "persistent":
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=settings,
                tenant=config.tenant,
                database=config.database
            )
            chroma = Chroma(
                collection_name=config.collection_name,
                embedding_function=embeddings_model,
                persist_directory=persist_dir,
                client=client
            )
        elif config.mode == "remote":
            conn_config = config.connection
            client = chromadb.HttpClient(
                host=conn_config.host,
                port=conn_config.port,
                ssl=conn_config.ssl,
                headers=conn_config.headers,
                settings=settings,
                tenant=config.tenant,
                database=config.database
            )
            chroma = Chroma(
                collection_name=config.collection_name,
                embedding_function=embeddings_model,
                client_settings=settings,
                client=client
            )
        else:
            raise NotImplementedError(f'{config.mode} for {type(config)} is not supported.')
        return chroma
