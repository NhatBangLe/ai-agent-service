from abc import abstractmethod

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.sessions import StdioConnection, StreamableHttpConnection

from src.config.configurer.interface import Configurer
from src.config.model.mcp import MCPConfiguration


class MCPConfigurer(Configurer):

    @abstractmethod
    def configure(self, config: MCPConfiguration, /, **kwargs):
        pass

    @abstractmethod
    async def async_configure(self, config: MCPConfiguration, /, **kwargs):
        pass

    @abstractmethod
    async def get_tools(self) -> list[BaseTool]:
        pass
