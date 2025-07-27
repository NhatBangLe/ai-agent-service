from pydantic import Field

from src.config.model.tool.search import SearchToolConfiguration, SearchToolType


class DuckDuckGoSearchToolConfiguration(SearchToolConfiguration):
    """
    A subclass of the SearchConfiguration class
    """
    type: SearchToolType = Field(default=SearchToolType.DUCKDUCKGO_SEARCH, frozen=True)

    def get_api_key_env(self) -> str | None:
        return None
