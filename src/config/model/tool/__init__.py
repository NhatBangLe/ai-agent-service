from abc import abstractmethod, ABC

from pydantic import Field

from src.config.model import Configuration


class ToolConfiguration(Configuration, ABC):
    """
    An interface for the tool configuration classes
    """
    name: str = Field(description="An unique name for determining tools purpose.", min_length=1, max_length=100)

    @abstractmethod
    def get_api_key_env(self) -> str | None:
        pass
