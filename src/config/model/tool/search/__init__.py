from pydantic import Field

from src.config.model.tool import ToolConfiguration

__all__ = ["SearchToolConfiguration", "brave", "duckduckgo"]


class SearchToolConfiguration(ToolConfiguration):
    """
    An interface for the search tool configuration classes
    """
    max_results: int = Field(alias="num_results", default=4)
