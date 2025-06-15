from pydantic import BaseModel, Field

__all__ = ["RecognizerConfiguration", "image"]


class RecognizerConfiguration(BaseModel):
    """
    An interface for recognizer configuration classes
    """
    enable: bool = True
    path: str = Field(description="Model file location")
    min_probability: float = Field(description="A low probability limit for specifying classes.", ge=0.0, le=1.0)
    max_results: int = Field(description="The maximum number of results recognized is used for prompting.",
                             default=4, ge=1, le=50)
    weight: float = Field(ge=0.0, le=1.0)
