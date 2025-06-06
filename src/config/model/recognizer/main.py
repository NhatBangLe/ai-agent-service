from pydantic import BaseModel, Field


class RecognizerConfiguration(BaseModel):
    """
    An interface for recognizer configuration classes
    """
    enable: bool = True
    path: str = Field(description="Model file location")
    min_probability: float = Field(description="A low probability limit for specifying classes.", ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
