from pydantic import BaseModel

from src.model.chat_model.main import LLMConfiguration
from src.model.prompt.main import PromptConfiguration
from src.model.recognizer.image.main import ImageRecognizerConfiguration
from src.model.retriever.main import RetrieverConfiguration
from src.model.tool.main import ToolConfiguration


class AgentConfiguration(BaseModel):
    """
    Agent configuration class for deserialize configuration files to pydantic object.
    """
    agent_name: str
    version: str | None = None
    description: str | None = None
    image_recognizer: ImageRecognizerConfiguration | None = None
    retrievers: list[RetrieverConfiguration] | None = None
    tools: list[ToolConfiguration] | None = None
    llm: LLMConfiguration
    prompt: PromptConfiguration
