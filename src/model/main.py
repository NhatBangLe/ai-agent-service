from pydantic import BaseModel

from src.model.chat_model.main import LLMConfiguration
from src.model.recognizer.main import RecognizerConfiguration
from src.model.retriever.main import RetrieverConfiguration
from src.model.tool.main import ToolConfiguration


class AgentConfiguration(BaseModel):
    """
    Agent configuration class for deserialize configuration files to pydantic object.
    """
    agent_name: str
    version: str | None = None
    description: str | None = None
    recognizers: list[RecognizerConfiguration] | None = None
    retrievers: list[RetrieverConfiguration] | None = None
    tools: list[ToolConfiguration] | None = None
    llm: LLMConfiguration
