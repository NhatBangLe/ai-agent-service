from typing import Optional, Annotated

from pydantic import BaseModel, Field

from model.chat_model.main import LLMConfiguration
from model.recognizer.main import RecognizerConfiguration
from model.retriever.main import RetrieverConfiguration


class AgentConfiguration(BaseModel):
    """
    Agent configuration class for deserialize configuration files to pydantic object.
    """
    agent_name: Annotated[str, Field(alias="name")]
    version: Optional[str] = None
    description: Optional[str] = None
    recognizers: Optional[list[RecognizerConfiguration]] = None
    retrievers: Optional[list[RetrieverConfiguration]] = None
    llm: LLMConfiguration