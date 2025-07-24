from src.config.model.tool.search import SearchToolConfiguration


class DuckDuckGoSearchToolConfiguration(SearchToolConfiguration):
    """
    A subclass of the SearchConfiguration class
    """

    def get_api_key_env(self) -> str | None:
        return None
