from pydantic import Field

from src.model.recognizer.main import RecognizerConfiguration


class ImageRecognizerConfiguration(RecognizerConfiguration):
    """

    """
    image_size: int = Field(ge=1)