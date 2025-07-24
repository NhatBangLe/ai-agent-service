from pydantic import Field

from src.config.model.tool.search import SearchToolConfiguration


class BraveSearchToolConfiguration(SearchToolConfiguration):
    """
    A subclass of the SearchConfiguration class
    """
    search_kwargs: dict | None = Field(description="Search arguments, see Brave Search API for more details",
                                       default=None)

    def get_api_key_env(self) -> str | None:
        return "BRAVE_API_KEY"
