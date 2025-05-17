from pydantic import Field

from model.tool.search.main import SearchToolConfiguration


class DuckDuckGoSearchToolConfiguration(SearchToolConfiguration):
    """
    A subclass of the SearchConfiguration class
    """
    max_results: int = Field(alias="num_results", default=4)