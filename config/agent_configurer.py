from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from model.agent import AgentConfiguration
from model.embeddings_model import EmbeddingsModelProvider
from model.vector_store import VectorStoreProvider

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
DEFAULT_PERSIST_DIRECTORY = "./langchain_db"


class AgentConfigurer:
    _config: AgentConfiguration
    _embeddings_model: Embeddings | None = None
    _vector_store: VectorStore | None = None
    _bm25_retriever: BM25Retriever | None = None
    _retriever: EnsembleRetriever | None = None

    def load_config(self):
        print(f'Loading configuration...')
        with open(DEFAULT_CONFIG_PATH, mode="r") as config_file:
            self._config = AgentConfiguration.model_validate_json(config_file.read())
            print(f'Done!')

    def configure_embeddings_model(self):
        emb_model_config = self._config.retriever.vector_store.embeddings_model
        embeddings_model_name = emb_model_config.model_name
        match emb_model_config.provider:
            case EmbeddingsModelProvider.HUGGING_FACE:
                self._embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_name)
            case EmbeddingsModelProvider.OLLAMA:
                self._embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    def configure_vector_store(self):
        print(f'Configuring vector store...')

        # Initialize Chroma vector store
        vs_config = self._config.retriever.vector_store
        match vs_config.provider:
            case VectorStoreProvider.CHROMA:
                self._vector_store = Chroma(
                    collection_name=vs_config.collection_name,
                    embedding_function=self._embeddings_model,
                    persist_directory=DEFAULT_PERSIST_DIRECTORY,  # Where to save data locally, remove if not necessary
                )

        # Add chunks to vector store
        print(f'Adding chunks to vector store...')
        chunk_ids = self._vector_store.add_documents(documents=doc_splits)
        print(len(chunk_ids))
        print(f'Done!')

    def configure_bm25(self):
        print(f'Configuring BM25 Index...')
        self._bm25_retriever = BM25Retriever.from_documents(documents=doc_splits)
        print(f'Done!')

    def configure_retriever(self):
        """
        Configure an ensemble retriever. This process will configure both vector store and BM25 index.
        You do not need to call the other configuring retriever functions.
        """
        self.configure_vector_store()
        self.configure_bm25()

        print(f'Configuring an ensemble retriever...')
        vs_retriever = self._vector_store.as_retriever()
        ensemble_weights = [0.5, 0.5]
        self._retriever = EnsembleRetriever(
            retrievers=[self._bm25_retriever, vs_retriever], weights=ensemble_weights
        )
        print(f'Done!')

    @property
    def retriever(self):
        return self._retriever
