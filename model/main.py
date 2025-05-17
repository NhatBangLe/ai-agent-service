from pydantic import BaseModel, Field

from model.chat_model.main import LLMConfiguration
from model.recognizer.main import RecognizerConfiguration
from model.retriever.main import RetrieverConfiguration
from model.tool.main import ToolConfiguration


class AgentConfiguration(BaseModel):
    """
    Agent configuration class for deserialize configuration files to pydantic object.
    """
    agent_name: str = Field(alias="name")
    version: str | None = None
    description: str | None = None
    recognizers: list[RecognizerConfiguration] | None = None
    retrievers: list[RetrieverConfiguration] | None = None
    tools: list[ToolConfiguration] | None = None
    llm: LLMConfiguration
