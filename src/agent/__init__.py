import datetime
from typing import TypedDict, Sequence, Literal
from langgraph.graph import MessagesState

__all__ = ["agent", "AgentStatus", "StateConfiguration", "Attachment", "ClassifiedAttachment", "State", "InputState"]

from pydantic import BaseModel, Field


class Attachment(TypedDict):
    id: str
    name: str
    mime_type: str
    save_path: str


class ClassifiedAttachment(Attachment):
    class_name: str
    probability: float


class StateConfiguration(TypedDict):
    """Configurable parameters for the agent."""
    my_configurable_param: str


class State(MessagesState):
    classified_attachments: list[ClassifiedAttachment] | None


class InputState(MessagesState):
    attachments: Sequence[Attachment] | None


class AgentStatus(BaseModel):
    status: Literal["ON", "OFF", "RESTART"] = Field(description="Current Agent status.")
    available_vector_stores: Sequence[str] = Field(description="A sequence contains names of available vector stores.")
    bm25_last_sync: datetime.datetime | None = Field(description="The last sync time of BM25 retriever.")
