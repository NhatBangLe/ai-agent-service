from abc import abstractmethod
from typing import Sequence

from langchain_core.tools import BaseTool

from src.config.configurer.interface import ToolConfigurer
from src.config.model.tool.search import SearchToolConfiguration


class SearchToolConfigurer(ToolConfigurer):

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_tools(self) -> Sequence[BaseTool]:
        pass

    @abstractmethod
    def get_tool(self, unique_name: str) -> BaseTool | None:
        pass

    @abstractmethod
    def get_config(self, unique_name: str) -> SearchToolConfiguration | None:
        pass
