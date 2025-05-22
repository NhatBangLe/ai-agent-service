from pydantic import BaseModel, Field, field_validator


# noinspection PyNestedDecorators
class PromptConfiguration(BaseModel):
    suggest_questions_prompt: str = Field(
        min_length=8,
        description="A prompt is used for instructing LLM to generate questions about specified topics",
        default="You are an expert at generating questions on various subjects."
                "\nHere are the topics; please provide relevant questions for each:"
                "\n{topics}")
    respond_prompt: str = Field(
        min_length=11,
        description="A prompt is used for instructing LLM to generate questions about specified topics",
        default="You are a Question-Answering assistant."
                "You are an AI Agent that built by using LangGraph and LLMs."
                "\nYour mission is that you need to analyze and answer questions."
                "\nAnswer by the language which is related to the questions."
                "\nYou can use accessible tools to retrieve more information if you want.")

    @field_validator("suggest_questions_prompt", mode="after")
    @classmethod
    def validate_suggest_questions_prompt(cls, prompt: str) -> str:
        placeholder = "{topics}"
        if not prompt.__contains__(placeholder):
            raise ValueError(f'Missing {placeholder} in suggest questions prompt.')
        return prompt