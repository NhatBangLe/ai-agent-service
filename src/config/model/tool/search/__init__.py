from abc import ABC
from enum import Enum

from pydantic import Field

from src.config.model.tool import ToolConfiguration


class SearchToolType(str, Enum):
    DUCKDUCKGO_SEARCH = "duckduckgo_search"


class SearchToolConfiguration(ToolConfiguration, ABC):
    """
    An interface for the search tool configuration classes
    """
    type: SearchToolType = Field(description="The type of search tool to use.")
    max_results: int = Field(ge=1, default=4)
