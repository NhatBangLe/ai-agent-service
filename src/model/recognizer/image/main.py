from typing import Any

from pydantic import Field

from src.model.recognizer.main import RecognizerConfiguration, Recognizer


class ImageRecognizerConfiguration(RecognizerConfiguration):
    """
    """
    image_size: int = Field(ge=1)


class ImageRecognizer(Recognizer):
    def predict(self, data: Any):
        pass
