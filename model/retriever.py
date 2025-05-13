from pydantic import BaseModel

from model.vector_store import VectorStoreConfiguration


class RetrieverConfiguration(BaseModel):
    use_external_searching: bool = True
    vector_store: VectorStoreConfiguration