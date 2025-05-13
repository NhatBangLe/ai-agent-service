from pydantic import BaseModel, Field


class ImageRecognizerConfiguration(BaseModel):
    """
    Image recognizer class for the recognizer.image property in configuration files.
    """
    use_image_recognizer: bool = True
    path: str = Field(description="Image Recognizer model path")


class RecognizerConfiguration(BaseModel):
    """
    Recognizer class for the recognizer property in configuration files.
    """
    image: ImageRecognizerConfiguration = Field(description="Image recognizer configuration")