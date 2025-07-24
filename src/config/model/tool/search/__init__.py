from abc import ABC

from pydantic import Field

from src.config.model.tool import ToolConfiguration


class SearchToolConfiguration(ToolConfiguration, ABC):
    """
    An interface for the search tool configuration classes
    """
    max_results: int = Field(alias="num_results", ge=1, default=4)
