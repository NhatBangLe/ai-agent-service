from abc import abstractmethod, ABC
from typing import TypedDict, Iterable

from pydantic import BaseModel, Field, field_validator

__all__ = ["image", "ClassDescriptor", "RecognizerOutput", "RecognizingResult", "Recognizer"]


class ClassDescriptor(BaseModel):
    name: str = Field(description="Name of data class", min_length=1)
    description: str = Field(description="Description for this class, use to search relevant information.",
                             min_length=10)


# noinspection PyNestedDecorators
class RecognizerOutput(BaseModel):
    is_configured: bool = Field(default=False, description="Whether the recognizer is configured")
    classes: list[ClassDescriptor] = Field(description="A list of data classes")

    @field_validator("classes", mode="after")
    @classmethod
    def remove_classes_duplicate(cls, classes: list[ClassDescriptor]):
        class_names = set()
        nodup_classes: list[ClassDescriptor] = []
        for data_class in classes:
            if data_class.name not in class_names:
                class_names.add(data_class.name)
                nodup_classes.append(data_class)
        return nodup_classes


class RecognizingResult(TypedDict):
    classes: Iterable[str]
    probabilities: Iterable[float] | None
    inference_time: float


class Recognizer(ABC):
    """
    An interface for using recognizers
    """

    @abstractmethod
    def predict(self, **kwargs) -> RecognizingResult:
        pass

    @abstractmethod
    async def async_predict(self, **kwargs) -> RecognizingResult:
        pass
