from typing import TypedDict, Literal
from uuid import UUID

from langgraph.graph import MessagesState


class Attachment(TypedDict):
    id: UUID
    mime_type: str


class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


class ClassifiedClass(TypedDict):
    data_type: Literal["image", "text"]
    class_name: str
    probability: float


class State(MessagesState):
    classified_classes: list[ClassifiedClass] | None


class InputState(MessagesState):
    attachments: list[Attachment] | None
