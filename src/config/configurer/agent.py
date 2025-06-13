import logging
import os
import string
from typing import cast, Sequence

import chromadb
import jsonpickle
from langchain.chat_models import init_chat_model
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import RetrieverLike
from langchain_core.tools import BaseTool, create_retriever_tool
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import TextSplitter
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row, DictRow
from sqlmodel import select

from src.config.model.chat_model.google_genai import GoogleGenAILLMConfiguration
from src.config.model.embeddings.hugging_face import HuggingFaceEmbeddingsConfiguration
from src.config.model.embeddings.main import EmbeddingsModelConfiguration
from src.config.model.main import AgentConfiguration
from src.config.model.retriever.bm25 import BM25Configuration
from src.config.model.retriever.vector_store.chroma import ChromaVSConfiguration
from src.config.model.retriever.vector_store.main import VectorStoreConfiguration
from src.config.model.tool.search.duckduckgo import DuckDuckGoSearchToolConfiguration
from src.config.model.tool.search.main import SearchToolConfiguration
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


class AgentConfigurer:
    _config: AgentConfiguration | None = None
    _embeddings_model: Embeddings | None = None
    _vector_stores: dict[str, VectorStore] = {}
    _bm25_retriever: BM25Retriever | None = None
    _tools: list[BaseTool] = []
    _llm: BaseChatModel | None = None
    _image_recognizer: ImageRecognizer | None = None
    _checkpointer: BaseCheckpointSaver | None
    _logger = logging.getLogger(__name__)

    async def configure(self):
        self._load_config()
        self._configure_llm()
        self._configure_tools()
        self._configure_retriever_tool()
        self._configure_image_recognizer()
        await self._configure_checkpointer()

    async def shutdown(self):
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
        self._config = AgentConfiguration.model_validate(jsonpickle.decode(json))
        self._logger.info(f'Loaded configuration successfully!')

    async def _configure_checkpointer(self):
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
        self._checkpointer = checkpointer

    def _configure_llm(self):
        """Configures the language model (LLM) for the agent.

        This method initializes the `self._llm` attribute based on the LLM
        configuration specified in the `self._config` object.

        Raises:
            RuntimeError: If the `AgentConfiguration` object (`self._config`)
                is None, indicating that the agent has not been properly configured.
            TypeError: If the LLM provider specified in the configuration is not
                currently supported.

        Returns:
            None
        """
        if self._config is None:
            raise RuntimeError("AgentConfiguration object is None.")

        config = self._config.llm
        if isinstance(config, GoogleGenAILLMConfiguration):
            genai = cast(GoogleGenAILLMConfiguration, config)
            self._llm = init_chat_model(
                model_provider=genai.provider,
                model=genai.model_name,
                temperature=genai.temperature,
                timeout=genai.timeout,
                max_tokens=genai.max_tokens,
                max_retries=genai.max_retries,
                top_p=genai.top_p,
                top_k=genai.top_k,
            )
        # elif isinstance(config, AnthropicLLMConfiguration):
        #     anthropic = typing.cast(AnthropicLLMConfiguration, config)
        #     self._llm = init_chat_model(
        #         model_provider=anthropic.provider,
        #         model=anthropic.model_name,
        #         temperature=anthropic.temperature,
        #         timeout=anthropic.timeout,
        #         stop=anthropic.stop_sequences,
        #         base_url=anthropic.base_url,
        #         max_tokens=anthropic.max_tokens,
        #         max_retries=anthropic.max_retries,
        #         top_p=anthropic.top_p,
        #         top_k=anthropic.top_k
        #     )
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
        #         top_k=ollama.top_k,
        #     )
        else:
            self._llm = None
            raise NotImplementedError(f'{config} is not supported.')

    def _configure_retriever_tool(self):
        """Configures and adds a retriever tool to the agent's available tools.

        This method iterates through the retriever configurations specified in
        `self._config.retrievers`, configures each retriever based on its type,
        and combines them into an `EnsembleRetriever`.
        Finally, it creates a Langchain retriever tool from
        the ensemble and adds it to the agent's `_tools` list.

        Raises:
            RuntimeError: If the `AgentConfiguration` object (`self._config`)
                is None, indicating that the agent has not been properly configured.
            NotImplementedError: If a retriever configuration type is encountered,
                that is not currently supported.

        Returns:
            None
        """
        if self._config is None:
            raise RuntimeError("AgentConfiguration object is None.")
        retriever_configs = self._config.retrievers
        if retriever_configs is None or len(retriever_configs) == 0:
            return

        retrievers: list[RetrieverLike] = []
        ensemble_weights = []
        for config in retriever_configs:
            if isinstance(config, VectorStoreConfiguration):
                vs_config = cast(VectorStoreConfiguration, config)
                vector_store = self._configure_vector_store(vs_config)
                search_kwargs = {
                    'fetch_k': vs_config.fetch_k,
                    'lambda_mult': vs_config.lambda_mult
                } if vs_config.search_type == "mmr" else {
                    'k': vs_config.k
                }
                self._vector_stores[config.name] = vector_store
                retrievers.append(vector_store.as_retriever(
                    search_type=vs_config.search_type,
                    search_kwargs=search_kwargs
                ))
            elif isinstance(config, BM25Configuration):
                self._configure_bm25(cast(BM25Configuration, config))
                retrievers.append(self._bm25_retriever)
            else:
                raise NotImplementedError(f'{type(config)} is not supported.')
            ensemble_weights.append(config.weight)

        retriever = EnsembleRetriever(retrievers=retrievers, weights=ensemble_weights)
        tool = create_retriever_tool(
            retriever,
            name="ensemble_information_retriever",
            description=(
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
                "If you cannot answer after using this tool, you can use another tool to retrieve more information."))
        self._tools.append(tool)

    def _configure_tools(self):
        tool_configs = self._config.tools
        if tool_configs is None or len(tool_configs) == 0:
            return

        self._tools = []
        for tool in tool_configs:
            if isinstance(tool, SearchToolConfiguration):
                search_tool = self._configure_search_tool(cast(SearchToolConfiguration, tool))
                self._tools.append(search_tool)
            else:
                raise NotImplementedError(f'{type(tool)} is not supported.')

    def _configure_vector_store(self, config: VectorStoreConfiguration):
        """Configures the vector store for storing and retrieving embeddings.

        This method initializes a vector store based on the provided `VectorStoreConfiguration`.

        Args:
            config: An instance of `VectorStoreConfiguration` containing the
                configuration parameters for the vector store.

        Raises:
            TypeError: If the vector store provider specified in the
                configuration is not currently supported.

        Returns:
            None
        """
        self._configure_embeddings_model(config.embeddings_model)
        self._logger.debug("Configuring vector store...")

        persist_dir = os.path.join(get_config_folder_path(), config.persist_directory)
        if isinstance(config, ChromaVSConfiguration):
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
                    embedding_function=self._embeddings_model,
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
                    embedding_function=self._embeddings_model,
                    client_settings=settings,
                    client=client
                )
            else:
                raise NotImplementedError(f'{config.mode} for {type(config)} is not supported.')

            vector_store = chroma
        else:
            raise NotImplementedError(f'{type(config)} is not supported.')

        self._logger.debug(f"Configured vector store {config.name} successfully.")
        return vector_store

    def _configure_embeddings_model(self, config: EmbeddingsModelConfiguration):
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
            None
        """
        self._logger.debug("Configuring embeddings model...")

        model_name = config.model_name
        if isinstance(config, HuggingFaceEmbeddingsConfiguration):
            self._embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
        else:
            self._embeddings_model = None
            raise NotImplementedError(f'{type(config)} is not supported.')

        self._logger.debug("Configured embeddings model successfully.")

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

        text_splitter = self.text_splitter
        chunks: list[Document] = []
        from src.data.database import create_session
        with create_session() as session:
            db_docs: Sequence[DBDocument] = session.exec(select(DBDocument)).all()
            for db_doc in db_docs:
                if db_doc.source == DocumentSource.UPLOADED:
                    doc_loader = get_document_loader(db_doc.save_path, db_doc.mime_type)
                    chunks += doc_loader.load_and_split(text_splitter)
                elif db_doc.source == DocumentSource.EXTERNAL and db_doc is not None:
                    store_name = db_doc.embed_to_vs
                    if store_name not in self._vector_stores:
                        self._logger.warning(f'Cannot use Document {db_doc.id} for the BM25 retriever. '
                                             f'Because vector store with name {store_name} '
                                             f'has not been configured yet.')
                        continue
                    chunk_ids = [chunk.id for chunk in db_doc.chunks]
                    vector_store = self._vector_stores[store_name]
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

        self._bm25_retriever = BM25Retriever.from_documents(documents=chunks,
                                                            preprocess_func=preprocess)
        self._logger.debug("Configured BM25 retriever successfully.")

    def _configure_search_tool(self, config: SearchToolConfiguration) -> BaseTool:
        self._logger.debug("Configuring search tool...")

        if isinstance(config, DuckDuckGoSearchToolConfiguration):
            duckduckgo = cast(DuckDuckGoSearchToolConfiguration, config)
            tool = DuckDuckGoSearchResults(name="duckduckgo_search",
                                           num_results=duckduckgo.max_results,
                                           output_format='list')
        # elif isinstance(config, BraveSearchToolConfiguration):
        #     brave = cast(BraveSearchToolConfiguration, config)
        #     return BraveSearch.from_api_key(api_key=brave.api_key, search_kwargs=brave.search_kwargs)
        else:
            raise NotImplementedError(f'{type(config)} is not supported.')

        self._logger.debug("Configured search tool successfully.")
        return tool

    def _configure_image_recognizer(self):
        self._logger.debug("Configuring image recognizer...")

        if self._config is None:
            raise RuntimeError("AgentConfiguration object is None.")

        config = self._config.image_recognizer
        if config is None or config.enable is False:
            self._logger.info("Image recognizer is disabled.")
            return

        max_workers = os.getenv("IMAGE_RECOGNIZER_MAX_WORKERS", "4")
        self._image_recognizer = ImageRecognizer(config=self._config.image_recognizer, max_workers=int(max_workers))
        self._image_recognizer.configure()

        self._logger.debug("Configured image recognizer successfully.")

    @property
    def text_splitter(self) -> TextSplitter:
        raise NotImplementedError

    @property
    def tools(self):
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
    def vector_stores(self):
        return self._vector_stores

    @property
    def bm25_retriever(self):
        return self._bm25_retriever

    @property
    def checkpointer(self):
        return self._checkpointer
