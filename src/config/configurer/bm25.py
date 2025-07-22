import asyncio
import datetime
import logging
import string
from pathlib import Path

from dependency_injector.wiring import inject
from docling.document_converter import DocumentConverter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from . import RetrieverConfigurer
from .embeddings import EmbeddingsConfigurer
from .vector_store import VectorStoreConfigurer
from ..model.retriever.bm25 import BM25Configuration
from ...data.base_model import DocumentSource
from ...provide import DocumentRepositoryProvide, FileServiceProvide
from ...util import TextPreprocessing
from ...util.function import get_config_folder_path, get_datetime_now


@inject
def _get_document_repository(repository: DocumentRepositoryProvide):
    return repository


@inject
async def _get_file_path_by_id(file_id: str, file_service: FileServiceProvide):
    file = await file_service.get_metadata_by_id(file_id)
    if file is None:
        return None
    return file.path


class BM25Configurer(RetrieverConfigurer):
    _config: BM25Configuration | None
    _retriever: BM25Retriever | None
    _last_sync: datetime.datetime | None
    _logger = logging.getLogger(__name__)

    def __init__(self):
        super().__init__()
        self._config = None
        self._retriever = None
        self._last_sync = None

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

        # Configures Embeddings model
        embeddings_configurer: EmbeddingsConfigurer = kwargs["embeddings_configurer"]
        embeddings_config = config.embeddings_model
        await embeddings_configurer.async_configure(embeddings_config)
        embeddings_model = embeddings_configurer.get_model(embeddings_config.name)
        if embeddings_model is None:
            raise ValueError(f'No {config.embeddings_model} embeddings model has configured yet.')

        chunks: list[Document] | None = None
        document_repository = _get_document_repository()
        db_docs = await document_repository.get_all_unembedded_vs()
        if len(db_docs) > 0:
            uploaded_docs = list(filter(lambda doc: doc.source == DocumentSource.UPLOADED, db_docs))

            async def get_from_uploaded_docs():
                self._logger.debug('Collecting chunks from uploaded documents.')
                get_path_tasks: list[asyncio.Task[str | None]] = []
                async with asyncio.TaskGroup() as group:
                    for doc in uploaded_docs:
                        get_path_tasks.append(group.create_task(_get_file_path_by_id(doc.file_id)))
                doc_paths: list[str] = list(filter(lambda p: p is not None, [t.result() for t in get_path_tasks]))

                converter = DocumentConverter()
                results = converter.convert_all(doc_paths)
                documents = [Document(page_content=result.document.export_to_markdown(),
                                      metadata={"source": doc_path, "total_pages": len(result.pages)})
                             for doc_path, result in zip(doc_paths, results)]
                return documents

            external_docs = list(filter(lambda doc: doc.source == DocumentSource.EXTERNAL, db_docs))

            async def get_from_external_docs():
                self._logger.debug('Collecting chunks from external documents.')
                async with asyncio.TaskGroup() as group:
                    doc_tasks: list[asyncio.Task[list[Document]]] = []
                    for doc in external_docs:
                        store_name = doc.embed_to_vs
                        vector_store = vs_configurer.get_store(store_name)
                        if vector_store is None:
                            self._logger.warning(f'Cannot use Document {doc.id} for the BM25 retriever. '
                                                 f'Because vector store with name {store_name} '
                                                 f'has not been configured yet.')
                            continue
                        chunk_ids = [chunk.id for chunk in doc.chunks]
                        doc_tasks.append(group.create_task(vector_store.aget_by_ids(chunk_ids)))
                documents: list[Document] = []
                for t in doc_tasks:
                    documents += t.result()
                return documents

            chunk_tasks: list[asyncio.Task[list[Document]]] = []
            async with asyncio.TaskGroup() as tg:
                if len(uploaded_docs) > 0:
                    chunk_tasks.append(tg.create_task(get_from_uploaded_docs()))
                if len(external_docs) > 0:
                    chunk_tasks.append(tg.create_task(get_from_external_docs()))

            # Chunking documents
            chunker = SemanticChunker(embeddings_model)
            docs: list[Document] = []
            for task in chunk_tasks:
                docs += task.result()
            chunks = chunker.split_documents(docs)

        if chunks is not None and len(chunks) > 0:
            removal_words_file_path = Path(get_config_folder_path(),
                                           config.removal_words_path) if config.removal_words_path else None
            helper = TextPreprocessing(removal_words_file_path) if removal_words_file_path else None

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
            self._config = config
            self._last_sync = get_datetime_now()
            self._logger.debug("Configured BM25 retriever successfully.")

            async def update_document_bm25_status():
                for doc in db_docs:
                    doc.embed_bm25 = True
                await document_repository.save_all(db_docs)

            asyncio.create_task(update_document_bm25_status())
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

    @property
    def config(self):
        return self._config
