from logging import Logger, getLogger

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import BaseTool

from src.config.configurer.interface.search_tool import SearchToolConfigurer
from src.config.model.tool.search import SearchToolConfiguration, SearchToolType


class SearchToolConfigurerImpl(SearchToolConfigurer):
    _tools: dict[str, tuple[SearchToolConfiguration, BaseTool]] | None = None
    _logger: Logger = getLogger(__name__)

    def configure(self, config, /, **kwargs):
        self._logger.debug("Configuring search tool...")
        if self._tools is None:
            self._tools = {}

        if config.type == SearchToolType.DUCKDUCKGO_SEARCH:
            tool = DuckDuckGoSearchResults(name=config.name,
                                           num_results=config.max_results,
                                           output_format='list')
            self._tools[config.name] = (config, tool)
        # elif isinstance(config, BraveSearchToolConfiguration):
        #     brave = cast(BraveSearchToolConfiguration, config)
        #     return BraveSearch.from_api_key(api_key=brave.api_key, search_kwargs=brave.search_kwargs)
        else:
            raise NotImplementedError(f'{type(config)} is not supported.')

        self._logger.debug("Configured search tool successfully.")

    async def async_configure(self, config, /, **kwargs):
        self.configure(config)

    def destroy(self, **kwargs):
        pass

    async def async_destroy(self, **kwargs):
        pass

    def get_tools(self):
        if self._tools is None:
            return []
        return [tool for _, (_, tool) in self._tools.items()]

    def get_tool(self, unique_name: str):
        if self._tools is None:
            return None
        value = self._tools[unique_name]
        return value[1] if value is not None else None

    def get_config(self, unique_name: str):
        if self._tools is None:
            return None
        value = self._tools[unique_name]
        return value[0] if value is not None else None
