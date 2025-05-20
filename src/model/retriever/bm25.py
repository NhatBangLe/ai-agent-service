from pydantic import Field

from src.model.retriever.main import RetrieverConfiguration

DEFAULT_BM25_PATH = "bm25.pkl"


class BM25Configuration(RetrieverConfiguration):
    path: str = Field(default=DEFAULT_BM25_PATH, min_length=1)
    # k: Annotated[int, Field(description="Number of documents to return.", default=4)]
    # use_tokenizer: Annotated[
    #     bool, Field(description="Use NLTK - Punkt Tokenizer Models to preprocess documents",
    #                 default=False)]
