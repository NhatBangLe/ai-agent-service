import asyncio
import datetime
import logging
import os
import string
from typing import Sequence

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sqlmodel import select

from src.config.configurer import RetrieverConfigurer
from src.config.configurer.embeddings import EmbeddingsConfigurer
from src.config.configurer.vector_store import VectorStoreConfigurer
from src.config.model.retriever.bm25 import BM25Configuration
from src.data.base_model import DocumentSource
from src.data.model import Document as DBDocument
from src.util import TextPreprocessing
from src.util.constant import DEFAULT_TIMEZONE
from src.util.function import get_document_loader, get_config_folder_path


class BM25Configurer(RetrieverConfigurer):
    _retriever: BM25Retriever | None = None
    _last_sync: datetime.datetime | None = None
    _logger = logging.getLogger(__name__)

    def configure(self, config: BM25Configuration, /, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_configure(config, **kwargs))

    async def async_configure(self, config: BM25Configuration, /, **kwargs):
        """
        Configures and initializes the BM25 retriever for efficient text search and retrieval.

        This method is responsible for gathering document chunks from various sources,
        configuring text preprocessing function, and then building the BM25 retriever instance
        based on the provided configuration.

        Functionality:
            1.  **Document Loading & Chunking**:
                Retrieves documents from the database.
                For `UPLOADED` documents, it loads and splits content into chunks using `text_splitter`.
                For `EXTERNAL` documents, it retrieves pre-existing chunks from a configured vector store,
                logging warnings if the store is not found.
            2.  **Text Preprocessing Setup**:
                Sets up a `TextPreprocessing` helper based on `config.removal_words_path`.
                Defines a `preprocess` function that converts text to lowercase, removes punctuation,
                optionally removes emojis and emoticons, and applies custom word removal,
                finally splitting the text into tokens.
            3.  **BM25 Retriever Initialization**:
                Initializes `self._bm25_retriever`, providing all collected chunks and the defined `preprocess` function.

        Args:
            config: An object containing the configuration settings for the BM25 retriever.

        Raises:
            ValueError: For unsupported document sources.
        """
        self._logger.debug("Configuring BM25 retriever...")
        vs_configurer: VectorStoreConfigurer | None = kwargs["vs_configurer"]
        if vs_configurer is None:
            raise ValueError(f'Configure BM25 Retriever must provide a VectorStoreConfigurer.')
        embeddings_configurer: EmbeddingsConfigurer | None = kwargs["embeddings_configurer"]
        if embeddings_configurer is None:
            raise ValueError(f'Configure BM25 Retriever must provide a EmbeddingsConfigurer.')

        embeddings_model = embeddings_configurer.get_model(config.embeddings_model)
        if embeddings_model is None:
            raise ValueError(f'No {config.embeddings_model} embeddings model has configured yet.')
        # text_splitter = self.text_splitter
        # chunker = SemanticChunker()

        chunks: list[Document] = []
        from src.data.database import create_session
        with create_session() as session:
            db_docs: Sequence[DBDocument] = session.exec(select(DBDocument)).all()
            for db_doc in db_docs:
                if db_doc.source == DocumentSource.UPLOADED:
                    doc_loader = get_document_loader(db_doc.save_path, db_doc.mime_type)
                    # chunks += doc_loader.load_and_split(text_splitter)
                elif db_doc.source == DocumentSource.EXTERNAL and db_doc is not None:
                    store_name = db_doc.embed_to_vs
                    vector_store = vs_configurer.get_store(store_name)
                    if vector_store is None:
                        self._logger.warning(f'Cannot use Document {db_doc.id} for the BM25 retriever. '
                                             f'Because vector store with name {store_name} '
                                             f'has not been configured yet.')
                        continue
                    chunk_ids = [chunk.id for chunk in db_doc.chunks]
                    chunks += await vector_store.aget_by_ids([str(chunk_id) for chunk_id in chunk_ids])
                else:
                    raise ValueError(f'Unsupported DocumentSource {db_doc.source}')

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

        if len(chunks) != 0:
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
