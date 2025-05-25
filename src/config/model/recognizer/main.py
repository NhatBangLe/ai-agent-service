from typing import Any

from pydantic import BaseModel, Field, field_validator


class ClassDescriptor(BaseModel):
    name: str = Field(description="Name of data class", min_length=1)
    description: str = Field(description="Description for this class, use to search information on Internet",
                             min_length=10)


# noinspection PyNestedDecorators
class RecognizerConfiguration(BaseModel):
    """
    An interface for recognizer configuration classes
    """
    enable: bool = True
    path: str = Field(description="Model file location")
    min_probability: float = Field(description="A low probability limit for specifying classes.", ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    classes: list[ClassDescriptor] = Field(min_length=1)

    @field_validator("classes", mode="before")
    @classmethod
    def remove_classes_duplicate(cls, classes: list[ClassDescriptor]):
        class_names = set()
        nodup_classes: list[ClassDescriptor] = []
        for data_class in classes:
            if data_class.name not in class_names:
                class_names.add(data_class.name)
                nodup_classes.append(data_class)
        return nodup_classes


class Recognizer(BaseModel):
    """
    An interface for using recognizers
    """

    def predict(self, data: Any) -> list[tuple[str, float]]:
        """
        Predict data.
        :param data: Based on each model.
        :return: Predicted `(data class, probability)` tuples.
        """
        raise NotImplementedError
