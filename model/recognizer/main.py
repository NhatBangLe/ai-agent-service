from enum import Enum

from pydantic import BaseModel, Field


class RecognizerType(str, Enum):
    """
    All supported recognizers.
    """
    IMAGE = "image"


class RecognizerConfiguration(BaseModel):
    """
    An interface for recognizer configuration classes
    """
    enable: bool = True
    path: str = Field(description="Model file location")
    type: RecognizerType = Field(description="All subclasses must specify this attribute.")
