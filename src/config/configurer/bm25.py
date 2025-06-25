import asyncio
import datetime
import logging
import os
import string
from typing import Sequence

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from sqlmodel import select

from src.config.configurer import RetrieverConfigurer
from src.config.configurer.embeddings import EmbeddingsConfigurer
from src.config.configurer.vector_store import VectorStoreConfigurer
from src.config.model.retriever.bm25 import BM25Configuration
from src.data.base_model import DocumentSource
from src.data.model import Document as DBDocument
from src.util import TextPreprocessing
from src.util.constant import DEFAULT_TIMEZONE
from src.util.function import get_documents, get_config_folder_path


class BM25Configurer(RetrieverConfigurer):
    _retriever: BM25Retriever | None = None
    _last_sync: datetime.datetime | None = None
    _logger = logging.getLogger(__name__)

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
        self._logger.debug("Configuring BM25 retriever...")
        if "vs_configurer" not in kwargs or not isinstance(kwargs["vs_configurer"], VectorStoreConfigurer):
            raise ValueError(f'Configure BM25 Retriever must provide a VectorStoreConfigurer.')
        if ("embeddings_configurer" not in kwargs or
                not isinstance(kwargs["embeddings_configurer"], EmbeddingsConfigurer)):
            raise ValueError(f'Configure BM25 Retriever must provide a EmbeddingsConfigurer.')

        vs_configurer: VectorStoreConfigurer = kwargs["vs_configurer"]
        embeddings_configurer: EmbeddingsConfigurer = kwargs["embeddings_configurer"]
        embeddings_model = embeddings_configurer.get_model(config.embeddings_model)
        if embeddings_model is None:
            raise ValueError(f'No {config.embeddings_model} embeddings model has configured yet.')

        chunks: list[Document] = []
        from src.data.database import create_session
        with create_session() as session:
            db_docs: Sequence[DBDocument] = session.exec(select(DBDocument)).all()
            if len(db_docs) > 0:
                chunker = SemanticChunker(embeddings_model)
                for db_doc in db_docs:
                    self._logger.debug(f'Collecting chunks from document with id {db_doc.id}.')

                    if db_doc.source == DocumentSource.UPLOADED:
                        documents = await get_documents(db_doc.save_path, db_doc.mime_type)
                        chunks += chunker.split_documents(documents)
                    elif db_doc.source == DocumentSource.EXTERNAL and db_doc is not None:
                        store_name = db_doc.embed_to_vs
                        vector_store = vs_configurer.get_store(store_name)
                        if vector_store is None:
                            self._logger.warning(f'Cannot use Document {db_doc.id} for the BM25 retriever. '
                                                 f'Because vector store with name {store_name} '
                                                 f'has not been configured yet.')
                            continue
                        chunk_ids = [chunk.id for chunk in db_doc.chunks]
                        chunks += await vector_store.aget_by_ids(chunk_ids)
                    else:
                        raise ValueError(f'Unsupported DocumentSource {db_doc.source}')

        if len(chunks) != 0:
            removal_words_file_path = os.path.join(get_config_folder_path(), config.removal_words_path)
            helper = TextPreprocessing(str(removal_words_file_path)) if removal_words_file_path else None

            def preprocess(text: str) -> list[str]:
                normalized_text = (text.lower()  # make to lower case
                                   .translate(str.maketrans('', '', string.punctuation)))  # remove punctuations
                if config.enable_remove_emoji:
                    normalized_text = TextPreprocessing.remove_emoji(normalized_text)
                if config.enable_remove_emoticon:
                    normalized_text = TextPreprocessing.remove_emoticons(normalized_text)
                if helper:
                    normalized_text = helper.remove_words(normalized_text)
                return normalized_text.split()

            self._logger.debug(f'Constructing BM25 retriever from retrieved chunks...')
            self._retriever = BM25Retriever.from_documents(documents=chunks,
                                                           preprocess_func=preprocess,
                                                           k=config.k)
            self._last_sync = datetime.datetime.now(DEFAULT_TIMEZONE)
            self._logger.debug("Configured BM25 retriever successfully.")
        else:
            self._logger.info("No chunks for initializing BM25 retriever. Skipping...")

    def destroy(self, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_destroy(**kwargs))

    async def async_destroy(self, **kwargs):
        pass

    @property
    def retriever(self):
        return self._retriever

    @property
    def last_sync(self):
        return self._last_sync
