import datetime
from typing import TypedDict, Sequence, Literal
from langgraph.graph import MessagesState

__all__ = ["agent", "AgentStatus", "StateConfiguration", "Attachment", "ClassifiedAttachment", "State", "InputState"]


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


class AgentStatus(TypedDict):
    status: Literal["ON", "OFF", "RESTART"]
    available_vector_stores: Sequence[str]
    bm25_last_sync: datetime.datetime | None
