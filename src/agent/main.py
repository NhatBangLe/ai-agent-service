"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""
import asyncio
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.state import State, InputState, Configuration, Attachment, ClassifiedClass
from src.config.main import AgentConfigurer

# Agent Configurer
configurer = AgentConfigurer()


def get_topics_from_classified_classes(classified_classes: list[ClassifiedClass]):
    image_class_descriptors = configurer.config.image_recognizer.classes

    topics: list[str] = []
    for classified_class in classified_classes:
        if classified_class["data_type"] == "image":
            found_topics = [descriptor.description for descriptor in image_class_descriptors
                            if descriptor.name == classified_class["class_name"]]
            topics.append(*found_topics)
        else:
            raise NotImplementedError(f'Classified class for text data has not been supported yet.')
    return topics


async def query_or_respond(state: State) -> State:
    llm = configurer.llm
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=configurer.config.prompt.respond_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])

    # Create prompt for the recognized topics
    classified_classes = state["classified_classes"]
    topic_prompt: str | None = None
    if classified_classes is not None:
        topics = get_topics_from_classified_classes(classified_classes)
        topic_prompt = ("The provided questions may be related to the following topics:"
                        f'{'\n'.join(topics)}') if len(topics) != 0 else None

    # Create real prompt to use
    messages = (state["messages"] + [SystemMessage(content=topic_prompt)]
                if topic_prompt is not None else state["messages"])
    prompt = prompt_template.invoke({
        "messages": messages
    })

    chat_model = llm.bind_tools(configurer.tools) if configurer.tools is not None else llm
    response = await chat_model.ainvoke(prompt)
    return {
        "messages": [response],
        "classified_classes": state["classified_classes"]
    }


async def suggest_questions(state: State) -> State:
    classified_classes = state["classified_classes"]
    topics: list[str] = get_topics_from_classified_classes(classified_classes)

    prompt = PromptTemplate.from_template(configurer.config.respond_prompt.suggest_questions_prompt)
    response = await configurer.llm.ainvoke(prompt.format(topics="\n".join(topics)))
    return {
        "messages": [response],
        "classified_classes": state["classified_classes"]
    }


async def classify_data(state: InputState) -> State:
    messages = state["messages"]
    image_recognizer = configurer.image_recognizer
    if image_recognizer is None:
        return {"classified_classes": None, "messages": messages}

    attachments = state["attachments"]
    images: list[Attachment] = [att for att in attachments if att["mime_type"].__contains__("image")]

    async def recognize_image(url: str):
        image = url
        predicted_classes = image_recognizer.predict(image)
        min_prob = configurer.config.image_recognizer.min_probability

        await asyncio.sleep(1)
        # return [accept_class for accept_class, prob in predicted_classes if prob >= min_prob]
        return [ClassifiedClass(data_type="image", class_name="ctu"),
                ClassifiedClass(data_type="image", class_name="change_later")]  # temporary

    recognize_image_tasks = [recognize_image(img["url"]) for img in images]
    try:
        topics: list[ClassifiedClass] = await asyncio.gather(*recognize_image_tasks)
    except Exception as e:
        print(f"\nCaught an unexpected error: {type(e).__name__}: {e}")
        topics = []

    return {"classified_classes": topics, "messages": messages}


def routes_condition(state: State) -> Literal["suggest_questions", "query_or_respond"]:
    latest_messages = state["messages"][-1]
    classified_classes = state["classified_classes"]

    if not isinstance(latest_messages, HumanMessage) and classified_classes is not None and len(
            classified_classes) != 0:
        return "suggest_questions"
    return "query_or_respond"


# Define the graph
def make_graph(config: RunnableConfig):
    configurer.configure()

    print(f'Building agent graph...')
    graph = StateGraph(state_schema=State, config_schema=Configuration, input=InputState)
    graph.add_node("query_or_respond", query_or_respond)
    graph.add_node("classify_data", classify_data)
    graph.add_node("suggest_questions", suggest_questions)

    if configurer.tools is not None:
        tools = configurer.tools
        graph.add_node("tools", ToolNode(tools))
        graph.add_conditional_edges("query_or_respond", tools_condition,
                                    {
                                        "tools": "tools",
                                        END: END,
                                    })
        graph.add_edge("tools", "query_or_respond")

    graph.set_entry_point("classify_data")
    graph.add_conditional_edges("classify_data", routes_condition, {
        "query_or_respond": "query_or_respond",
        "suggest_questions": "suggest_questions"
    })
    graph.add_edge("suggest_questions", END)
    print(f'Done!')

    return graph.compile(name=configurer.config.agent_name)
