from pydantic import BaseModel, Field

__all__ = ["PromptConfiguration"]


class PromptConfiguration(BaseModel):
    suggest_questions_prompt: str = Field(
        min_length=8,
        description="A prompt is used for instructing LLM to generate questions about specified topics",
        default="You are an expert at generating questions on various subjects.")
    respond_prompt: str = Field(
        min_length=11,
        description="A prompt is used for instructing LLM to generate questions about specified topics",
        default="You are a Question-Answering assistant."
                "You are an AI Agent that built by using LangGraph and LLMs."
                "\nYour mission is that you need to analyze and answer questions."
                "\nAnswer by the language which is related to the questions."
                "\nYou can use accessible tools to retrieve more information if you want.")
