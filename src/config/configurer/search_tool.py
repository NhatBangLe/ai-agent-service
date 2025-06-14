from logging import Logger, getLogger
from typing import Sequence

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import BaseTool

from src.config.configurer import ToolConfigurer
from src.config.model.tool.search import SearchToolConfiguration
from src.config.model.tool.search.duckduckgo import DuckDuckGoSearchToolConfiguration


class SearchToolConfigurer(ToolConfigurer):
    _tools: dict[str, tuple[SearchToolConfiguration, BaseTool]] | None = None
    _logger: Logger = getLogger(__name__)

    def configure(self, config: SearchToolConfiguration, /, **kwargs):
        """
        Configures and registers a search tool based on the provided configuration.

        This method inspects the type of `config` to determine which search tool
        to initialize. The configured tool is then stored internally for later use.

        Args:
            config: The configuration object for the search tool. This object
                    specifies the type of search tool to be set up and includes
                    all necessary parameters for its initialization.

        Raises:
            NotImplementedError:
                If the `config` type is not supported. Currently, only
                `DuckDuckGoSearchToolConfiguration` is supported.
        """
        self._logger.debug("Configuring search tool...")
        if self._tools is None:
            self._tools = {}

        if isinstance(config, DuckDuckGoSearchToolConfiguration):
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

    async def async_configure(self, config: SearchToolConfiguration, /, **kwargs):
        """
        Async-configures and registers a search tool based on the provided configuration.

        This method inspects the type of `config` to determine which search tool
        to initialize. The configured tool is then stored internally for later use.

        Args:
            config: The configuration object for the search tool. This object
                    specifies the type of search tool to be set up and includes
                    all necessary parameters for its initialization.

        Raises:
            NotImplementedError:
                If the `config` type is not supported. Currently, only
                `DuckDuckGoSearchToolConfiguration` is supported.
        """
        self.configure(config)

    def destroy(self, **kwargs):
        pass

    async def async_destroy(self, **kwargs):
        pass

    def get_tools(self) -> Sequence[BaseTool]:
        if self._tools is None:
            return []
        return [tool for _, (_, tool) in self._tools.items()]

    def get_tool(self, unique_name: str) -> BaseTool | None:
        if self._tools is None:
            return None
        value = self._tools[unique_name]
        return value[1] if value is not None else None

    def get_config(self, unique_name: str) -> SearchToolConfiguration | None:
        if self._tools is None:
            return None
        value = self._tools[unique_name]
        return value[0] if value is not None else None
