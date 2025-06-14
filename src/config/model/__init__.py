from pydantic import BaseModel

__all__ = ["agent", "retriever", "recognizer", "chat_model", "embeddings", "tool", "prompt", "Configuration"]


class Configuration(BaseModel):
    pass
