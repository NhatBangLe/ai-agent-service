from pydantic import Field

from src.config.model import Configuration

__all__ = ["search", "ToolConfiguration"]


class ToolConfiguration(Configuration):
    """
    An interface for the tool configuration classes
    """
    name: str = Field(description="An unique name for determining tools purpose.", min_length=1)
