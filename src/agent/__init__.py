from typing import TypedDict, Literal, Sequence

from langgraph.graph import MessagesState

__all__ = ["agent", "StateConfiguration", "ClassifiedClass", "State", "InputState"]


class StateConfiguration(TypedDict):
    """Configurable parameters for the agent."""
    my_configurable_param: str


class ClassifiedClass(TypedDict):
    data_type: Literal["image", "text"]
    class_name: str
    probability: float


class State(MessagesState):
    classified_classes: list[ClassifiedClass] | None


class InputState(MessagesState):
    image_paths: Sequence[str] | None  # Images need to be recognized to determine classes.
