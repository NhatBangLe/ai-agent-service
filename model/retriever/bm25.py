from typing import Annotated
from pydantic import Field

from model.retriever.main import RetrieverType, RetrieverConfiguration


class BM25Configuration(RetrieverConfiguration):
    type = RetrieverType.BM25
    k: Annotated[int, Field(description="Number of documents to return.", default=4)]
    use_tokenizer: Annotated[bool, Field(description="Use NLTK - Punkt Tokenizer Models to preprocess documents", default=False)]
