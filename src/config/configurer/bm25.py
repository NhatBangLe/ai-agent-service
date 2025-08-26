import asyncio
import datetime
import logging
import string
from pathlib import Path

from dependency_injector.wiring import inject
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .interface.bm25 import BM25Configurer
from ..model.retriever.bm25 import BM25Configuration
from ...provide import DocumentRepositoryProvide
from ...util import TextPreprocessing
from ...util.function import get_config_folder_path, get_datetime_now


@inject
def _get_document_repository(repository: DocumentRepositoryProvide):
    return repository


class BM25ConfigurerImpl(BM25Configurer):
    _config: BM25Configuration | None
    _text_preprocessing_helper: TextPreprocessing | None
    _retriever: BM25Retriever | None
    _last_sync: datetime.datetime | None
    _logger = logging.getLogger(__name__)

    def __init__(self):
        super().__init__()
        self._config = None
        self._retriever = None
        self._last_sync = None
        self._text_preprocessing_helper = None

    def configure(self, config, vs_configurer, /, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_configure(config, vs_configurer, **kwargs))

    async def async_configure(self, config, vs_configurer, /, **kwargs):
        self._logger.info("Configuring BM25 retriever...")
        self._config = config
        self._text_preprocessing_helper = self._get_text_preprocessing_helper()

        document_repository = _get_document_repository()
        db_docs = await document_repository.get_all_chunks()
        if len(db_docs.keys()) <= 0:
            self._logger.info("No document for initializing BM25 retriever. Skipping...")
            return

        chunks: list[Document] = []
        for vs_name in db_docs.keys():
            store = vs_configurer.get_store(vs_name)
            chunk_ids = db_docs[vs_name]
            if store is not None and len(chunk_ids) != 0:
                retrieved_chunks = await store.aget_by_ids(chunk_ids)
                chunks += retrieved_chunks

        self._logger.debug(f'Constructing BM25 retriever from chunks...')
        self._retriever = BM25Retriever.from_documents(documents=chunks,
                                                       preprocess_func=lambda x: self._preprocess(x),
                                                       k=self._config.k)
        self._logger.info("Configured BM25 retriever successfully.")

        self._last_sync = get_datetime_now()

    def destroy(self, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_destroy(**kwargs))

    async def async_destroy(self, **kwargs):
        pass

    async def add_documents(self, documents):
        if self._retriever is None:
            self._logger.warning(
                "Cannot add documents to BM25 retriever because BM25 retriever has not been configured yet.")
            return
        retriever: BM25Retriever = self._retriever
        new_docs = retriever.docs + documents
        self._retriever = BM25Retriever.from_documents(documents=new_docs,
                                                       preprocess_func=retriever.preprocess_func,
                                                       k=retriever.k)

    async def replace_all_documents(self, documents):
        if self._retriever is None:
            self._logger.warning(
                "Cannot replace all documents in BM25 retriever because BM25 retriever has not been configured yet.")
            return
        retriever: BM25Retriever = self._retriever
        self._retriever = BM25Retriever.from_documents(documents=documents,
                                                       preprocess_func=retriever.preprocess_func,
                                                       k=retriever.k)

    def _get_text_preprocessing_helper(self):
        removal_words_file_path = Path(get_config_folder_path(),
                                       self._config.removal_words_path) if self._config.removal_words_path else None
        return TextPreprocessing(removal_words_file_path) if removal_words_file_path else None

    def _preprocess(self, text: str) -> list[str]:
        normalized_text = (text.lower()  # make to lower case
                           .translate(str.maketrans('', '', string.punctuation)))  # remove punctuations
        if self._config.enable_remove_emoji:
            normalized_text = TextPreprocessing.remove_emoji(normalized_text)
        if self._config.enable_remove_emoticon:
            normalized_text = TextPreprocessing.remove_emoticons(normalized_text)
        if self._text_preprocessing_helper:
            normalized_text = self._text_preprocessing_helper.remove_words(normalized_text)
        return normalized_text.split()

    @property
    def retriever(self):
        return self._retriever

    @property
    def last_sync(self):
        return self._last_sync

    @property
    def config(self):
        return self._config
