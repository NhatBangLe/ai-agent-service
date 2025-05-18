from langgraph.graph.graph import CompiledGraph

from src.config.main import AgentConfigurer


class Agent:
    _compiled_graph: CompiledGraph
    _configurer: AgentConfigurer

    def __init__(self):
        self._configurer = AgentConfigurer()
