from typing import Literal

from pydantic import Field

from src.config.model import Configuration
from src.config.model.recognizer import RecognizerConfiguration


class ImagePreprocessingConfiguration(Configuration):
    """
    An interface for pre-processing image subclasses.
    """


class ImageRecognizerConfiguration(RecognizerConfiguration):
    """
    An interface for image recognizer subclasses.
    """
    device: Literal["auto", "cpu", "cuda"] = 'auto'
    preprocessing: list[ImagePreprocessingConfiguration] | None = Field(default=None)
