from langchain_core.tools import create_retriever_tool
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode

from config.main import AgentConfigurer


class Agent:
    _compiled_graph: CompiledGraph
    _configurer: AgentConfigurer

    def __init__(self):
        self._configurer = AgentConfigurer()


    def build_graph(self):
        print(f'Building agent graph...')
        retriever_tool = create_retriever_tool(
            self._configurer.retriever,
            "retrieve_blog_posts",
            "Search and return information about Lilian Weng blog posts.",
        )
        tools = [retriever_tool] + self._configurer.tools

        graph = StateGraph()
        graph.add_node(ToolNode(tools))

        self._compiled_graph = graph.compile()
        print(f'Done!')


    def run(self):
        self._configurer.configure()
        self.build_graph()

