from typing import Callable

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from nltk.tokenize import word_tokenize
from langchain.retrievers import EnsembleRetriever
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers.bm25 import default_preprocessing_func
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from model.chat_model.main import LLMProvider
from model.embeddings.main import EmbeddingsModelConfiguration, EmbeddingsModelProvider
from model.main import AgentConfiguration
from model.retriever.bm25 import BM25Configuration
from model.retriever.main import RetrieverType
from model.retriever.vector_store.main import VectorStoreConfiguration, VectorStoreProvider, DEFAULT_PERSIST_DIRECTORY

import nltk

# Load and split documents to chunks
print(f'Loading documents...')
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=100, chunk_overlap=50
# )
# pdf_loader = PyPDFLoader("../resource/CFG.pdf")
# doc_splits = pdf_loader.load_and_split(text_splitter)
doc_splits = []
print(f'Done!')

DEFAULT_CONFIG_PATH = "./config.json"


class AgentConfigurer:
    _config: AgentConfiguration | None = None
    _embeddings_model: Embeddings | None = None
    _vector_store: VectorStore | None = None
    _bm25_retriever: BM25Retriever | None = None
    _retriever: EnsembleRetriever | None = None
    _llm: BaseChatModel | None = None

    def load_config(self):
        """Loads the agent configuration from the default configuration file.

        This method reads the JSON content from the file specified by
        `DEFAULT_CONFIG_PATH` and validates it against the `AgentConfiguration`
        Pydantic model. The loaded configuration is then stored in the
        `self._config` attribute.

        Prints messages to the console indicating the start and completion
        of the configuration loading process.

        Raises:
            FileNotFoundError: If the `DEFAULT_CONFIG_PATH` does not exist.
            pydantic.ValidationError: If the content of the configuration file
                does not conform to the structure defined by the
                `AgentConfiguration` model.
            Exception: For other potential errors during file reading.

        Returns:
            None
        """
        print(f'Loading configuration...')
        with open(DEFAULT_CONFIG_PATH, mode="r") as config_file:
            self._config = AgentConfiguration.model_validate_json(config_file.read())
            print(f'Done!')

    def configure_llm(self):
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
        match config.provider:
            case LLMProvider.ANTHROPIC:
                self._llm = ChatAnthropic(model_name=config.model_name, timeout=10, stop=None)
            case _:
                self._llm = None
                raise TypeError(f'LLM provider {config.provider} is not supported.')

    def configure_retriever(self):
        """Configures the document retriever for the agent.

        This method initializes the `self._retriever` attribute, which can be
        an ensemble of different retriever types (e.g., vector store, BM25,
        external searching). It iterates through the retriever configurations
        specified in `self._config.retrievers`, configures each retriever based
        on its type, and combines them into an `EnsembleRetriever` with the
        specified weights.

        Raises:
            RuntimeError: If the `AgentConfiguration` object (`self._config`)
                is None, indicating that the agent has not been properly configured.
            TypeError: If the retriever type specified in the configuration is
                not currently supported or if the type matching fails.

        Returns:
            None
        """
        if self._config is None:
            raise RuntimeError("AgentConfiguration object is None.")

        retrievers: list[Runnable[str, list[Document]]] = []
        ensemble_weights = []
        for retriever in self._config.retrievers:
            match retriever.type:
                case RetrieverType.VECTOR_STORE:
                    self.configure_vector_store(retriever)
                    retrievers.append(self._vector_store.as_retriever())
                case RetrieverType.BM25:
                    self.configure_bm25(retriever)
                    retrievers.append(self._bm25_retriever)
                case RetrieverType.EXTERNAL_SEARCHING:
                    self.configure_external_searching()
                case _:
                    raise TypeError(f'Retriever type {retriever.type} is not supported.')
            ensemble_weights.append(retriever.weight)

        self._retriever = EnsembleRetriever(retrievers=retrievers, weights=ensemble_weights)

    def configure_vector_store(self, config: VectorStoreConfiguration):
        """Configures the vector store for storing and retrieving embeddings.

        This method initializes the `self._vector_store` attribute based on the
        provided `VectorStoreConfiguration`.

        Args:
            config: An instance of `VectorStoreConfiguration` containing the
                configuration parameters for the vector store.

        Raises:
            TypeError: If the vector store provider specified in the
                configuration is not currently supported.

        Returns:
            None
        """
        match config.provider:
            case VectorStoreProvider.CHROMA:
                self._vector_store = Chroma(
                    collection_name=config.collection_name,
                    embedding_function=self._embeddings_model,
                    persist_directory=DEFAULT_PERSIST_DIRECTORY,  # Where to save data locally, remove if not necessary
                )
            case _:
                self._vector_store = None
                raise TypeError(f'Vector store provider {config.provider} is not supported.')

        if self._vector_store is not None:
            # Add chunks to vector store
            print(f'Adding chunks to vector store...')
            chunk_ids = self._vector_store.add_documents(documents=doc_splits)
            print(len(chunk_ids))

    def configure_embeddings_model(self, config: EmbeddingsModelConfiguration):
        """Configures the embeddings model for text embedding generation.

        This method initializes the `self._embeddings_model` attribute based on
        the provided `EmbeddingsModelConfiguration`.

        Args:
            config: An instance of `EmbeddingsModelConfiguration` containing the
                configuration parameters for the embeddings model.

        Raises:
            TypeError: If the embeddings model provider specified in the
                configuration is not currently supported.

        Returns:
            None
        """
        embeddings_model_name = config.model_name
        match config.provider:
            case EmbeddingsModelProvider.HUGGING_FACE:
                self._embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_name)
            case EmbeddingsModelProvider.OLLAMA:
                self._embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_name)
            case _:
                self._embeddings_model = None
                raise TypeError(f'Embeddings model provider {config.provider} is not supported.')

    def configure_bm25(self, config: BM25Configuration):
        """Configures the BM25 retriever for document retrieval.

        This method initializes the BM25 retriever (`self._bm25_retriever`)
        based on the provided `BM25Configuration`. It handles the preprocessing
        of documents using either the default preprocessing function or the
        `word_tokenize` function from NLTK if specified in the configuration.

        Args:
            config: An instance of `BM25Configuration` containing the
                configuration parameters for the BM25 retriever.

        Raises:
            nltk.downloader.DownloadError: If `config.use_tokenizer` is True
                and the 'punkt_tab' resource for NLTK is not downloaded.

        Returns:
            None
        """
        preprocess_func: Callable[[str], list[str]] = default_preprocessing_func
        if config.use_tokenizer is True:
            nltk.download("punkt_tab")
            preprocess_func = word_tokenize
        self._bm25_retriever = BM25Retriever.from_documents(documents=doc_splits, preprocess_func=preprocess_func)

    def configure_external_searching(self):
        raise NotImplementedError("External searching is not supported.")

    @property
    def retriever(self):
        return self._retriever
