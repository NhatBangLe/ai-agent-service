import asyncio
import logging
import os
from typing import cast, Sequence

import jsonpickle
from langchain.chat_models import init_chat_model
from langchain.retrievers import EnsembleRetriever
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import RetrieverLike
from langchain_core.tools import BaseTool, create_retriever_tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row, DictRow

from src.config.configurer import Configurer
from src.config.configurer.bm25 import BM25Configurer
from src.config.configurer.embeddings import EmbeddingsConfigurer
from src.config.configurer.search_tool import SearchToolConfigurer
from src.config.configurer.vector_store import VectorStoreConfigurer
from src.config.model.agent import AgentConfiguration
from src.config.model.chat_model import LLMConfiguration
from src.config.model.chat_model.google_genai import GoogleGenAILLMConfiguration, convert_safety_settings_to_genai
from src.config.model.recognizer.image import ImageRecognizerConfiguration
from src.config.model.retriever import RetrieverConfiguration
from src.config.model.retriever.bm25 import BM25Configuration
from src.config.model.retriever.vector_store import VectorStoreConfiguration
from src.config.model.tool import ToolConfiguration
from src.config.model.tool.search import SearchToolConfiguration
from src.process.recognizer.image.main import ImageRecognizer
from src.util.function import get_config_folder_path


def _get_config_file_path():
    config_file_name = "config.json"
    config_path = os.path.join(get_config_folder_path(), config_file_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Missing {config_file_name} file in {config_path}')
    return config_path


class AgentConfigurer(Configurer):
    _config: AgentConfiguration | None = None
    _bm25_configurer: BM25Configurer | None = None
    _embeddings_configurer: EmbeddingsConfigurer | None = None
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

        self._embeddings_configurer = EmbeddingsConfigurer()
        for config in self._config.embeddings:
            await self._embeddings_configurer.async_configure(config)

        # Configure tools
        tools = self._configure_tools(self._config.tools)
        ensemble_tool = await self._configure_retriever_tool(self._config.retrievers)
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

    async def _configure_retriever_tool(self, configs: Sequence[RetrieverConfiguration]) -> BaseTool | None:
        if configs is None or len(configs) == 0:
            return None

        retrievers: list[RetrieverLike] = []
        ensemble_weights = []

        if self._vs_configurer is None:  # init for using at the fist time
            self._vs_configurer = VectorStoreConfigurer()
        vs_configs = list(filter(lambda c: isinstance(c, VectorStoreConfiguration), configs))
        for config in vs_configs:
            if isinstance(config, VectorStoreConfiguration):
                vs_config = cast(VectorStoreConfiguration, config)
                await self._vs_configurer.async_configure(vs_config,
                                                          embeddings_configurer=self._embeddings_configurer)
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
            else:
                raise NotImplementedError(f'{type(config)} is not supported.')

        bm25_configs: list[BM25Configuration] = list(filter(lambda c: isinstance(c, BM25Configuration), configs))
        if len(bm25_configs) != 0:
            config = bm25_configs[0]
            if self._bm25_configurer is None:  # init for using at the fist time
                self._bm25_configurer = BM25Configurer()
            await self._bm25_configurer.async_configure(config,
                                                        vs_configurer=self._vs_configurer,
                                                        embeddings_configurer=self._embeddings_configurer)
            retriever = self._bm25_configurer.retriever
            if retriever is not None:
                retrievers.append(retriever)
                ensemble_weights.append(config.weight)

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
    def checkpointer(self):
        return self._checkpointer

    @property
    def bm25_configurer(self):
        return self._bm25_configurer
