"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from typing import Any, Dict, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from src.config.main import AgentConfigurer

# Agent Configurer
configurer = AgentConfigurer()


class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


async def call_model(state: MessagesState, config: RunnableConfig) -> Dict[str, Any]:
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    llm_with_tools = configurer.llm
    if configurer.tools is not None:
        llm_with_tools = llm_with_tools.bind_tools(configurer.tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Define the graph
def make_graph(config: RunnableConfig):
    configurer.configure()

    print(f'Building agent graph...')
    graph = StateGraph(state_schema=MessagesState, config_schema=Configuration)
    graph.add_node(call_model)
    graph.add_edge(START, "call_model")

    if configurer.tools is not None:
        tools = configurer.tools
        graph.add_node("retrieve", ToolNode(tools))
        graph.add_conditional_edges("call_model",
                                    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
                                    tools_condition,
                                    {
                                        # Translate the condition outputs to nodes in our graph
                                        "tools": "retrieve",
                                        END: END,
                                    })
        graph.add_edge("retrieve", "call_model")
    print(f'Done!')

    return graph.compile(name=configurer.config.agent_name)
