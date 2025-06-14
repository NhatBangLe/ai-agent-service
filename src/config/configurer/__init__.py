from abc import ABC, abstractmethod

__all__ = ["Configurer", "RetrieverConfigurer", "ToolConfigurer", "agent", "vector_store"]

from src.config.model import Configuration


class Configurer(ABC):

    @abstractmethod
    def configure(self, config: Configuration, /, **kwargs):
        pass

    @abstractmethod
    async def async_configure(self, config: Configuration, /, **kwargs):
        pass

    @abstractmethod
    def destroy(self, **kwargs):
        pass

    @abstractmethod
    async def async_destroy(self, **kwargs):
        pass


class RetrieverConfigurer(Configurer, ABC):
    pass


class ToolConfigurer(Configurer, ABC):
    pass
