import asyncio
import logging
import os
from pathlib import Path
from typing import cast, Sequence

import jsonpickle
from dependency_injector.wiring import inject
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import RetrieverLike
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row, DictRow

from . import Configurer
from .bm25 import BM25Configurer
from .embeddings import EmbeddingsConfigurer
from .ensemble import EnsembleRetrieverConfigurer
from .mcp import MCPConfigurer
from .recognizer.image import ImageRecognizerConfigurer
from .search_tool import SearchToolConfigurer
from .vector_store import VectorStoreConfigurer
from ..model.agent import AgentConfiguration
from ..model.chat_model import ChatModelConfiguration
from ..model.chat_model.google_genai import GoogleGenAIChatModelConfiguration, convert_safety_settings_to_genai
from ..model.retriever.bm25 import BM25Configuration
from ..model.retriever.vector_store import VectorStoreConfiguration
from ..model.tool import ToolConfiguration
from ..model.tool.search import SearchToolConfiguration
from ...provide import DatabaseConnectionProvide, LabelServiceProvide, DocumentServiceProvide
from ...util.function import get_config_folder_path


def _get_config_file_path():
    config_file_name = "config.json"
    config_path = os.path.join(get_config_folder_path(), config_file_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Missing {config_file_name} file in {config_path}')
    return config_path


@inject
async def _configure_checkpointer(connection: DatabaseConnectionProvide):
    url = connection.get_url()
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


@inject
async def _insert_predefined_output_classes(output_config_path: Path, label_service: LabelServiceProvide):
    await label_service.insert_predefined_output_classes(output_config_path)


@inject
async def _insert_external_document(store_name: str, ext_data_file_path: Path, doc_service: DocumentServiceProvide):
    await doc_service.insert_external_document(store_name, ext_data_file_path)


class AgentConfigurer(Configurer):
    _config: AgentConfiguration | None
    _bm25_configurer: BM25Configurer | None
    _embeddings_configurer: EmbeddingsConfigurer | None
    _vs_configurer: VectorStoreConfigurer | None
    _search_configurer: SearchToolConfigurer | None
    _ensemble_configurer: EnsembleRetrieverConfigurer | None
    _mcp_configurer: MCPConfigurer | None
    _image_recognizer_configurer: ImageRecognizerConfigurer | None
    _tools: list[BaseTool]
    _llm: BaseChatModel | None
    _checkpointer: BaseCheckpointSaver | None
    _logger = logging.getLogger(__name__)

    def __init__(self):
        super().__init__()
        self._config = None
        self._bm25_configurer = None
        self._embeddings_configurer = EmbeddingsConfigurer()
        self._vs_configurer = None
        self._search_configurer = None
        self._ensemble_configurer = EnsembleRetrieverConfigurer()
        self._mcp_configurer = None
        self._image_recognizer_configurer = None
        self._tools = []
        self._llm = None
        self._checkpointer = None

    def configure(self, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_configure(**kwargs))

    async def async_configure(self, **kwargs):
        self._config = await self._load_config()
        self._llm = self._configure_llm(self._config.llm)

        # Configure retrievers
        if self._config.retrievers is not None and len(self._config.retrievers) != 0:
            retrievers: list[RetrieverLike] = []
            weights: list[float] = []

            vs_configs = list(filter(lambda c: isinstance(c, VectorStoreConfiguration), self._config.retrievers))
            async with asyncio.TaskGroup() as tg:
                if len(vs_configs) > 0:
                    for cf in vs_configs:
                        path = cf.external_data_config_path
                        if path is not None:
                            config_file_path = Path(get_config_folder_path(), path)
                            tg.create_task(_insert_external_document(store_name=cf.name,
                                                                     ext_data_file_path=config_file_path))
                    vs_task = tg.create_task(self._configure_vector_stores(vs_configs))
            if vs_task is not None:
                vs_retrievers, vs_weights = vs_task.result()
                retrievers += vs_retrievers
                weights += vs_weights

            bm25_configs = list(filter(lambda c: isinstance(c, BM25Configuration), self._config.retrievers))
            if len(bm25_configs) != 0:
                result = await self._configure_bm25(bm25_configs[0])
                if result is not None:
                    bm25_retriever, bm25_weight = result
                    retrievers.append(bm25_retriever)
                    weights.append(bm25_weight)

            await self._ensemble_configurer.async_configure(retrievers=retrievers, weights=weights)

        # Configure tools
        tools = []
        if self._config.tools is not None:
            configured_tools = self._configure_tools(self._config.tools)
            if configured_tools is not None:
                tools += configured_tools
        if self._ensemble_configurer.tool is not None:
            tools.append(self._ensemble_configurer.tool)
        if self._config.mcp is not None:
            self._mcp_configurer = MCPConfigurer()
            await self._mcp_configurer.async_configure(self._config.mcp)
            tools += await self._mcp_configurer.get_tools()
        if self._config.image_recognizer is not None:
            img_rec_config = self._config.image_recognizer
            output_config_path = Path(get_config_folder_path(), img_rec_config.output_config_path)
            self._image_recognizer_configurer = ImageRecognizerConfigurer()
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._image_recognizer_configurer.async_configure(img_rec_config))
                tg.create_task(_insert_predefined_output_classes(output_config_path))
            tools.append(self.image_recognizer_configurer.tool)
        self._tools = tools

        # Configure checkpointer
        self._checkpointer = await _configure_checkpointer()

    def destroy(self, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_destroy(**kwargs))

    async def async_destroy(self, **kwargs):
        if isinstance(self._checkpointer, AsyncPostgresSaver) is not None:
            checkpointer = cast(AsyncPostgresSaver, self._checkpointer)
            await checkpointer.conn.close()

    async def _load_config(self):
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

        def read_config_file():
            path = Path(config_file_path)
            json = path.read_text()
            return jsonpickle.decode(json)

        obj = await asyncio.to_thread(read_config_file)
        return AgentConfiguration.model_validate(obj)

    def _configure_llm(self, config: ChatModelConfiguration) -> BaseChatModel:
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
        if isinstance(config, GoogleGenAIChatModelConfiguration):
            genai = cast(GoogleGenAIChatModelConfiguration, config)
            safety_settings = convert_safety_settings_to_genai(genai.safety_settings) if genai.safety_settings else None
            llm = init_chat_model(
                model_provider="google_genai",
                model=genai.model_name,
                temperature=genai.temperature,
                timeout=genai.timeout,
                max_tokens=genai.max_tokens,
                max_retries=genai.max_retries,
                top_p=genai.top_p,
                top_k=genai.top_k,
                safety_settings=safety_settings)
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

    async def _configure_vector_stores(self, configs: Sequence[VectorStoreConfiguration]):
        retrievers: list[RetrieverLike] = []
        weights: list[float] = []
        if configs is None or len(configs) == 0:
            return retrievers, weights

        if self._vs_configurer is None:  # init for using at the fist time
            self._vs_configurer = VectorStoreConfigurer()
        for config in configs:
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
                weights.append(config.weight)
            else:
                raise NotImplementedError(f'{type(config)} is not supported.')

        return retrievers, weights

    async def _configure_bm25(self, config: BM25Configuration):
        if self._bm25_configurer is None:  # init for using at the fist time
            self._bm25_configurer = BM25Configurer()
        await self._bm25_configurer.async_configure(config,
                                                    vs_configurer=self._vs_configurer,
                                                    embeddings_configurer=self._embeddings_configurer)
        retriever = self._bm25_configurer.retriever
        if retriever is None:
            return None
        return retriever, config.weight

    def _configure_tools(self, configs: Sequence[ToolConfiguration]) -> Sequence[BaseTool] | None:
        if len(configs) == 0:
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

    @property
    def tools(self) -> Sequence[BaseTool]:
        return self._tools

    @property
    def chat_model(self) -> BaseChatModel:
        tools = self._tools
        llm = self._llm
        if llm is None:
            raise ValueError("Chat Model has not been configured yet.")
        return llm.bind_tools(tools=tools) if tools is not None else llm

    @property
    def config(self):
        return self._config

    @property
    def image_recognizer(self):
        if self._image_recognizer_configurer is None:
            return None
        return self._image_recognizer_configurer.image_recognizer

    @property
    def vector_store_configurer(self):
        return self._vs_configurer

    @property
    def checkpointer(self):
        return self._checkpointer

    @property
    def bm25_configurer(self):
        return self._bm25_configurer

    @property
    def image_recognizer_configurer(self):
        return self._image_recognizer_configurer
