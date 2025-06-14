import asyncio
import logging
import os
import string
from typing import cast, Sequence

import jsonpickle
from langchain.chat_models import init_chat_model
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import RetrieverLike
from langchain_core.tools import BaseTool, create_retriever_tool
from langchain_text_splitters import TextSplitter
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row, DictRow
from sqlmodel import select

from src.config.configurer import Configurer
from src.config.configurer.search_tool import SearchToolConfigurer
from src.config.configurer.vector_store import VectorStoreConfigurer
from src.config.model.agent import AgentConfiguration
from src.config.model.chat_model.google_genai import GoogleGenAILLMConfiguration, convert_safety_settings_to_genai
from src.config.model.chat_model.main import LLMConfiguration
from src.config.model.recognizer.image.main import ImageRecognizerConfiguration
from src.config.model.retriever import RetrieverConfiguration
from src.config.model.retriever.bm25 import BM25Configuration
from src.config.model.retriever.vector_store import VectorStoreConfiguration
from src.config.model.tool import ToolConfiguration
from src.config.model.tool.search import SearchToolConfiguration
from src.data.base_model import DocumentSource
from src.data.model import Document as DBDocument
from src.process.recognizer.image.main import ImageRecognizer
from src.util.function import get_config_folder_path, get_document_loader
from src.util.main import TextPreprocessing


def _get_config_file_path():
    config_file_name = "config.json"
    config_path = os.path.join(get_config_folder_path(), config_file_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Missing {config_file_name} file in {config_path}')
    return config_path


class AgentConfigurer(Configurer):
    _config: AgentConfiguration | None = None
    _bm25_retriever: BM25Retriever | None = None
    _vs_configurer: VectorStoreConfigurer | None = None
    _search_configurer: SearchToolConfigurer | None = None
    _tools: list[BaseTool] | None = None
    _llm: BaseChatModel | None = None
    _image_recognizer: ImageRecognizer | None = None
    _checkpointer: BaseCheckpointSaver | None
    _logger = logging.getLogger(__name__)

    ENSEMBLE_RETRIEVER_DESCRIPTION = (
        "A highly robust and comprehensive tool designed to retrieve the most relevant and "
        "accurate information from a vast knowledge base by combining multiple advanced search algorithms."
        "**USE THIS TOOL WHENEVER THE USER ASKS A QUESTION REQUIRING EXTERNAL KNOWLEDGE,"
        "FACTUAL INFORMATION, CURRENT EVENTS, OR DATA BEYOND YOUR INTERNAL TRAINING.**"
        "**Examples of when to use this tool:**"
        "- \"the capital of France?\""
        "- \"the history of the internet.\""
        "- \"the latest developments in AI?\""
        "- \"quantum entanglement.\""
        "**Crucially, use this tool for any query that cannot be answered directly from your"
        "pre-trained knowledge, especially if it requires up-to-date, specific, or detailed factual data.**"
        "The tool takes a single, concise search query as input."
        "If you cannot answer after using this tool, you can use another tool to retrieve more information.")

    def configure(self, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_configure(**kwargs))

    async def async_configure(self, **kwargs):
        self._config = self._load_config()
        self._llm = self._configure_llm(self._config.llm)

        # Configure tools
        tools = self._configure_tools(self._config.tools)
        ensemble_tool = self._configure_retriever_tool(self._config.retrievers)
        if tools is not None or ensemble_tool is not None:
            self._tools = []
            if tools is not None:
                self._tools += tools
            if ensemble_tool is not None:
                self._tools.append(ensemble_tool)

        self._image_recognizer = self._configure_image_recognizer(self._config.image_recognizer)
        self._checkpointer = await self._configure_checkpointer()

    def destroy(self, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_destroy(**kwargs))

    async def async_destroy(self, **kwargs):
        if isinstance(self._checkpointer, AsyncPostgresSaver) is not None:
            checkpointer = cast(AsyncPostgresSaver, self._checkpointer)
            await checkpointer.conn.close()

    def _load_config(self):
        """
        Loads the agent configuration from the configuration file.

        This method reads the JSON content from the config file and
        validates it against the `AgentConfiguration` Pydantic model.
        The loaded configuration is then stored in the `self._config` attribute.

        Raises:
            FileNotFoundError: If the `DEFAULT_CONFIG_PATH` does not exist.
            pydantic.ValidationError: If the content of the configuration file
                does not conform to the structure defined by the `AgentConfiguration` model.
            Exception: For other potential errors during file reading.

        Returns:
            None
        """
        config_file_path = _get_config_file_path()
        self._logger.info(f'Loading configuration...')
        with open(config_file_path, mode="r") as config_file:
            json = config_file.read()
        return AgentConfiguration.model_validate(jsonpickle.decode(json))

    @staticmethod
    async def _configure_checkpointer():
        from ...data.database import url
        conn_str = f"postgresql://{url.username}:{url.password}@{url.host}:{url.port}/{url.database}"
        conn = await AsyncConnection.connect(
            conninfo=conn_str,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row
        )
        checkpointer = AsyncPostgresSaver(conn=cast(AsyncConnection[DictRow], conn))

        await checkpointer.setup()
        return checkpointer

    def _configure_llm(self, config: LLMConfiguration) -> BaseChatModel:
        """Configures the language model (LLM) for the agent.

        This method configures the chat model.

        Args:
            config: The LLM configuration object.

        Raises:
            NotImplementedError: If the LLM provider specified in the configuration
                                is not currently supported.

        Returns:
            None
        """
        if isinstance(config, GoogleGenAILLMConfiguration):
            genai = cast(GoogleGenAILLMConfiguration, config)
            llm = init_chat_model(
                model_provider=genai.provider,
                model=genai.model_name,
                temperature=genai.temperature,
                timeout=genai.timeout,
                max_tokens=genai.max_tokens,
                max_retries=genai.max_retries,
                top_p=genai.top_p,
                top_k=genai.top_k,
                safety_settings=convert_safety_settings_to_genai(genai.safety_settings))
        # elif isinstance(config, OllamaLLMConfiguration):
        #     ollama = typing.cast(OllamaLLMConfiguration, config)
        #     self._llm = init_chat_model(
        #         model_provider=ollama.provider,
        #         model=ollama.model_name,
        #         temperature=ollama.temperature,
        #         seed=ollama.seed,
        #         num_ctx=ollama.num_ctx,
        #         num_predict=ollama.num_predict,
        #         repeat_penalty=ollama.repeat_penalty,
        #         stop=ollama.stop,
        #         top_p=ollama.top_p,
        #         top_k=ollama.top_k)
        else:
            raise NotImplementedError(f'{config} is not supported.')
        return llm

    def _configure_retriever_tool(self, configs: Sequence[RetrieverConfiguration]) -> BaseTool | None:
        """Configures and adds a retriever tool to the agent's available tools.

        This method iterates through the retriever configurations specified in
        `self._config.retrievers`, configures each retriever based on its type,
        and combines them into an `EnsembleRetriever`.
        Finally, it creates a Langchain retriever tool from
        the ensemble and adds it to the agent's `_tools` list.

        Args:
            configs: A `RetrieverConfiguration` sequence provides configurations.

        Raises:
            RuntimeError: If the `AgentConfiguration` object (`self._config`)
                is None, indicating that the agent has not been properly configured.
            NotImplementedError: If a retriever configuration type is encountered,
                that is not currently supported.

        Returns:
            None
        """
        if configs is None or len(configs) == 0:
            return None

        retrievers: list[RetrieverLike] = []
        ensemble_weights = []
        for config in configs:
            if isinstance(config, VectorStoreConfiguration):
                if self._vs_configurer is None:  # init for using at the fist time
                    self._vs_configurer = VectorStoreConfigurer()

                vs_config = cast(VectorStoreConfiguration, config)
                self._vs_configurer.configure(vs_config)
                search_kwargs = {
                    'fetch_k': vs_config.fetch_k,
                    'lambda_mult': vs_config.lambda_mult
                } if vs_config.search_type == "mmr" else {
                    'k': vs_config.k
                }
                vector_store = self._vs_configurer.get_store(vs_config.name)
                retrievers.append(vector_store.as_retriever(
                    search_type=vs_config.search_type,
                    search_kwargs=search_kwargs
                ))
                ensemble_weights.append(config.weight)
            elif isinstance(config, BM25Configuration):
                self._configure_bm25(cast(BM25Configuration, config))
                if self._bm25_retriever is not None:
                    retrievers.append(self._bm25_retriever)
                    ensemble_weights.append(config.weight)
            else:
                raise NotImplementedError(f'{type(config)} is not supported.')

        if len(retrievers) == 0:
            return None
        retriever = EnsembleRetriever(retrievers=retrievers, weights=ensemble_weights)
        return create_retriever_tool(
            retriever,
            name="ensemble_retriever",
            description=self.ENSEMBLE_RETRIEVER_DESCRIPTION)

    def _configure_tools(self, configs: Sequence[ToolConfiguration]) -> Sequence[BaseTool] | None:
        if configs is None or len(configs) == 0:
            return None

        tools: list[BaseTool] = []
        for config in configs:
            if isinstance(config, SearchToolConfiguration):
                if self._search_configurer is None:  # init for using at the fist time
                    self._search_configurer = SearchToolConfigurer()

                self._search_configurer.configure(config)
                search_tool = self._search_configurer.get_tool(config.name)
                tools.append(search_tool)
            else:
                raise NotImplementedError(f'{type(config)} is not supported.')

        return tools if len(tools) != 0 else None

    def _configure_bm25(self, config: BM25Configuration):
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

        # text_splitter = self.text_splitter
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
                    vector_store = self._vs_configurer.get_store(store_name)
                    if vector_store is None:
                        self._logger.warning(f'Cannot use Document {db_doc.id} for the BM25 retriever. '
                                             f'Because vector store with name {store_name} '
                                             f'has not been configured yet.')
                        continue
                    chunk_ids = [chunk.id for chunk in db_doc.chunks]
                    chunks += vector_store.get_by_ids([str(chunk_id) for chunk_id in chunk_ids])
                else:
                    raise ValueError(f'Unsupported DocumentSource {db_doc.source}')

        removal_words_file_path = os.path.join(get_config_folder_path(), config.removal_words_path)
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

        if len(chunks) != 0:
            self._bm25_retriever = BM25Retriever.from_documents(documents=chunks,
                                                                preprocess_func=preprocess)
            self._logger.debug("Configured BM25 retriever successfully.")
        else:
            self._logger.info("No chunks for initializing BM25 retriever. Skipping...")

    def _configure_image_recognizer(self, config: ImageRecognizerConfiguration) -> ImageRecognizer | None:
        self._logger.debug("Configuring image recognizer...")

        if config is None or config.enable is False:
            self._logger.info("Image recognizer is disabled.")
            return None

        max_workers = os.getenv("IMAGE_RECOGNIZER_MAX_WORKERS", "4")
        recognizer = ImageRecognizer(config=self._config.image_recognizer, max_workers=int(max_workers))
        recognizer.configure()

        self._logger.debug("Configured image recognizer successfully.")
        return recognizer

    @property
    def text_splitter(self) -> TextSplitter:
        raise NotImplementedError

    @property
    def tools(self) -> Sequence[BaseTool] | None:
        return self._tools

    @property
    def llm(self):
        return self._llm

    @property
    def config(self):
        return self._config

    @property
    def image_recognizer(self):
        return self._image_recognizer

    @property
    def vector_store_configurer(self):
        return self._vs_configurer

    @property
    def bm25_retriever(self):
        return self._bm25_retriever

    @property
    def checkpointer(self):
        return self._checkpointer
