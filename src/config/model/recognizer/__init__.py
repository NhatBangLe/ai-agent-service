from pydantic import Field

from src.config.model import Configuration


class RecognizerConfiguration(Configuration):
    """
    An interface for recognizer configuration classes
    """
    enable: bool = Field(default=True)
    path: str = Field(description="Model file location")
    min_probability: float = Field(description="A low probability limit for specifying classes.", ge=0.0, le=1.0)
    max_results: int = Field(description="The maximum number of results recognized is used for prompting.",
                             default=4, ge=1, le=50)
    output_config_path: str = Field(description="Path to an output configuration file.")
