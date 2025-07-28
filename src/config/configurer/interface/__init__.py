from abc import ABC, abstractmethod


class Configurer(ABC):

    @abstractmethod
    def configure(self, **kwargs):
        pass

    @abstractmethod
    async def async_configure(self, **kwargs):
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
