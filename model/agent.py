from typing import Optional

from pydantic import BaseModel

from model.llm import LLMConfiguration
from model.recognizer import RecognizerConfiguration
from model.retriever import RetrieverConfiguration


class AgentConfiguration(BaseModel):
    """
    Agent configuration class for deserialize configuration files to pydantic object.
    """
    agent_name: str
    version: Optional[str] = None
    description: Optional[str] = None
    recognizer: RecognizerConfiguration
    retriever: RetrieverConfiguration
    llm: LLMConfiguration