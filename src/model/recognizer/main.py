from pydantic import BaseModel, Field

class RecognizerConfiguration(BaseModel):
    """
    An interface for recognizer configuration classes
    """
    enable: bool = True
    path: str = Field(description="Model file location")
