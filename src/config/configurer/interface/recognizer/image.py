from abc import abstractmethod

from langchain_core.tools import BaseTool

from src.config.model.recognizer.image import ImageRecognizerConfiguration
from src.process.recognizer.image import ImageRecognizer
from .. import Configurer


class ImageRecognizerConfigurer(Configurer):

    @abstractmethod
    def configure(self, config: ImageRecognizerConfiguration, **kwargs):
        pass

    @abstractmethod
    async def async_configure(self, config: ImageRecognizerConfiguration, **kwargs):
        pass

    @property
    @abstractmethod
    def tool(self) -> BaseTool | None:
        pass

    @property
    @abstractmethod
    def image_recognizer(self) -> ImageRecognizer | None:
        pass
