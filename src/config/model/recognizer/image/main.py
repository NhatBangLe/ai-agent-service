from typing import Literal

from pydantic import Field, BaseModel

from src.config.model.recognizer.main import RecognizerConfiguration


class ImagePreprocessingConfiguration(BaseModel):
    """
    An interface for pre-processing image subclasses.
    """


# noinspection PyNestedDecorators
class ImageRecognizerConfiguration(RecognizerConfiguration):
    """
    An interface for image recognizer subclasses.
    """
    device: Literal["auto", "cpu", "cuda"] = 'auto'
    preprocessing: list[ImagePreprocessingConfiguration] | None = Field(default=None)
    output_config_path: str
