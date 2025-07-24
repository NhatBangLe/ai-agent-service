from pydantic import BaseModel, Field


class PromptConfiguration(BaseModel):
    respond_prompt: str = Field(
        min_length=1,
        description="A prompt is used for instructing large language models to generate responses to user input.",
        default="You are a Question-Answering assistant."
                "You are an AI Agent that built by using LangGraph and LLMs."
                "\nYour mission is that you need to analyze and answer questions."
                "\nAnswer by the language which is related to the questions."
                "\nYou can use accessible tools to retrieve more information if you want.")
