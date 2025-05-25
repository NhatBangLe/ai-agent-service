from pydantic import Field

from src.config.model.tool.search.main import SearchToolConfiguration


class BraveSearchToolConfiguration(SearchToolConfiguration):
    """
    A subclass of the SearchConfiguration class
    """
    api_key: str = Field(description="Must provide", min_length=1)
    search_kwargs: dict | None = Field(description="Search arguments, see Brave Search API for more details",
                                       default=None)
