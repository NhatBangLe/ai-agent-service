import asyncio
from logging import Logger, getLogger
from os import path
from typing import Sequence

import chromadb
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from src.config.configurer import Configurer
from src.config.model.embeddings import EmbeddingsConfiguration
from src.config.model.embeddings.hugging_face import HuggingFaceEmbeddingsConfiguration
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
        self._logger.debug(f"Configuring vector store {config.name}...")
        if self._vector_stores is None:
            self._vector_stores = {}

        if isinstance(config, ChromaVSConfiguration):
            chroma = self._configure_chroma(config)
            self._vector_stores[config.name] = (config, chroma)
        else:
            raise NotImplementedError(f'{type(config)} is not supported.')

        self._logger.debug(f"Configured vector store {config.name} successfully.")

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
        self.configure(config, **kwargs)

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
            self._logger.debug("No stores has been configured yet.")
            return None
        value = self._vector_stores[unique_name]
        return value[0] if value is not None else None

    def get_all_stores(self) -> Sequence[VectorStore]:
        if self._vector_stores is None:
            return []
        return [store for _, (_, store) in self._vector_stores.items()]

    def _configure_chroma(self, config: ChromaVSConfiguration):
        persist_dir = path.join(get_config_folder_path(), config.persist_directory)
        settings = chromadb.Settings(anonymized_telemetry=False)
        embeddings_model = self._configure_embeddings_model(config.embeddings_model)

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

    def _configure_embeddings_model(self, config: EmbeddingsConfiguration):
        """Configures the embedding model for text embedding generation.

        This method initializes the `self._embeddings_model` attribute based on
        the provided `EmbeddingsModelConfiguration`.

        Args:
            config: An instance of `EmbeddingsModelConfiguration` containing the
                configuration parameters for the embedding model.

        Raises:
            TypeError: If the embedding model provider specified in the
                configuration is not currently supported.

        Returns:
            The configured embeddings model.
        """
        self._logger.debug("Configuring embeddings model...")

        model_name = config.model_name
        if isinstance(config, HuggingFaceEmbeddingsConfiguration):
            model = HuggingFaceEmbeddings(model_name=model_name)
        else:
            raise NotImplementedError(f'{type(config)} is not supported.')

        self._logger.debug("Configured embeddings model successfully.")
        return model
