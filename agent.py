from langchain_core.tools import create_retriever_tool
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph

from config.main import AgentConfigurer


class Agent:
    graph: CompiledGraph
    configurer: AgentConfigurer

    def __init__(self):
        self.configurer = AgentConfigurer()
        self.configurer.load_config()


    def configure(self):
        print(f'Configuring agent...')
        self.configurer.configure_retriever()
        self.configurer.configure_llm()
        print(f'Done!')


    def build_graph(self):
        print(f'Building agent graph...')
        retriever_tool = create_retriever_tool(
            self.configurer.retriever,
            "retrieve_blog_posts",
            "Search and return information about Lilian Weng blog posts.",
        )

        graph = StateGraph()

        print(f'Done!')


    def run(self):
        self.configure()
        self.build_graph()

