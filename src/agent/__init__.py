import datetime
from typing import TypedDict, Sequence, Literal

from langgraph.graph import MessagesState

__all__ = ["agent", "AgentMetadata", "StateConfiguration", "Attachment", "State"]

from pydantic import BaseModel, Field


class Attachment(BaseModel):
    id: str = Field(description="Unique identifier of the attachment.", min_length=1)
    name: str = Field(description="Name of the attachment.", min_length=1)
    mime_type: str = Field(description="MIME type of the attachment.", min_length=1)
    path: str = Field(description="Path to the attachment.")


class StateConfiguration(TypedDict):
    pass


class State(MessagesState):
    attachment: Attachment | None


AgentStatus = Literal["ON", "OFF", "RESTART", "EMBED_DOCUMENT", "BM25_SYNC"]


class AgentMetadata(BaseModel):
    status: AgentStatus = Field(description="Current Agent status.")
    available_vector_stores: Sequence[str] = Field(description="A sequence contains names of available vector stores.")
    bm25_last_sync: datetime.datetime | None = Field(description="The last sync time of BM25 retriever.")
