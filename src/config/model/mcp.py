from enum import Enum
from typing import Literal

from pydantic import Field

from src.config.model import Configuration


class MCPTransport(str, Enum):
    STREAMABLE_HTTP = "streamable_http"
    STDIO = "stdio"


class MCPConnectionConfiguration(Configuration):
    type: MCPTransport = Field(description="The type of connection to use.")


class StreamableConnectionConfiguration(MCPConnectionConfiguration):
    type: MCPTransport = Field(default=MCPTransport.STREAMABLE_HTTP)

    url: str
    """The URL of the endpoint to connect to."""

    headers: dict[str, str] | None = Field(default=None)
    """HTTP headers to send to the endpoint."""

    timeout: int = Field(ge=0, default=30)
    """HTTP timeout (in seconds)."""

    sse_read_timeout: int = Field(ge=0, default=60 * 5)
    """How long (in seconds) the client will wait for a new event before disconnecting.
    All other HTTP operations are controlled by `timeout`."""

    terminate_on_close: bool = Field(default=True)
    """Whether to terminate the session on close."""

    # auth: NotRequired[httpx.Auth]
    # """Optional authentication for the HTTP client."""


class StdioConnectionConfiguration(MCPConnectionConfiguration):
    type: MCPTransport = Field(default=MCPTransport.STDIO)

    command: str
    """The executable to run to start the server."""

    args: list[str]
    """Command line arguments to pass to the executable."""

    env: dict[str, str] | None
    """The environment to use when spawning the process."""

    cwd: str | None
    """The working directory to use when spawning the process."""

    encoding: str = Field(default="utf-8")
    """The text encoding used when sending/receiving messages to the server."""

    encoding_error_handler: Literal["strict", "ignore", "replace"]
    """
    The text encoding error handler.

    See https://docs.python.org/3/library/codecs.html#codec-base-classes for
    explanations of possible values.
    """


class MCPConfiguration(Configuration):
    connections: dict[str, MCPConnectionConfiguration]
    """A dictionary mapping server names to connection configurations."""
