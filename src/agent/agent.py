import asyncio
import logging
from logging import Logger
from typing import Literal, Any, Sequence
from uuid import uuid4, UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import StateSnapshot

from src.agent import StateConfiguration, ClassifiedClass, State, InputState
from src.config.configurer.agent import AgentConfigurer
from src.util import FileInformation, Progress
from src.util.function import get_document_loader, get_topics_from_classified_classes


def _routes_condition(state: State) -> Literal["suggest_questions", "query_or_respond"]:
    latest_messages = state["messages"][-1]
    classified_classes = state["classified_classes"]

    if not isinstance(latest_messages, HumanMessage) and classified_classes is not None and len(
            classified_classes) != 0:
        return "suggest_questions"
    return "query_or_respond"


def _convert_topics_to_str(topics: Sequence[tuple[ClassifiedClass, str]]):
    return [f'Topic: {desc} - Probability: {classified_class["probability"]}'
            for classified_class, desc in topics]


class Agent:
    _status: Literal["ON", "OFF", "RESTART"]
    _configurer: AgentConfigurer
    _graph: CompiledStateGraph | None
    _is_configured: bool
    _logger: Logger

    def __init__(self, configurer: AgentConfigurer):
        self._status = "ON"
        self._configurer = configurer
        self._graph = None
        self._is_configured = False
        self._logger = logging.getLogger(__name__)

    def stream(self, input_msg: InputState, config: RunnableConfig | None = None, *,
               stream_mode: Literal["values", "updates", "messages"] | None = None):
        """
        Args:
            input_msg: The input to the graph.
            config: The configuration to use for the run.
            stream_mode: The mode to stream output, defaults to `self.stream_mode`.
                Options are:

                - `"values"`: Emit all values in the state after each step, including interrupts.
                    When used with functional API, values are emitted once at the end of the workflow.
                - `"updates"`: Emit only the node or task names and updates returned by the nodes or tasks after each step.
                    If multiple updates are made in the same step (e.g., multiple nodes are run), then those updates are emitted separately.
                - `"messages"`: Emit LLM messages token-by-token together with metadata for any LLM invocations inside nodes or tasks.
                    Will be emitted as 2-tuples `(LLM token, metadata)`.

                You can pass a list as the `stream_mode` parameter to stream multiple modes at once.
                The streamed outputs will be tuples of `(mode, data)`.
        """
        self.check_graph_available()

        graph: CompiledStateGraph = self._graph
        for state in graph.stream(input_msg, config, stream_mode=stream_mode):
            yield state

    async def astream(self, input_msg: InputState, config: RunnableConfig | None = None, *,
                      stream_mode: Literal["values", "updates", "messages"] | None = None):
        """
        Args:
            input_msg: The input to the graph.
            config: The configuration to use for the run.
            stream_mode: The mode to stream output, defaults to `self.stream_mode`.
                Options are:

                - `"values"`: Emit all values in the state after each step, including interrupts.
                    When used with functional API, values are emitted once at the end of the workflow.
                - `"updates"`: Emit only the node or task names and updates returned by the nodes or tasks after each step.
                    If multiple updates are made in the same step (e.g., multiple nodes are run), then those updates are emitted separately.
                - `"messages"`: Emit LLM messages token-by-token together with metadata for any LLM invocations inside nodes or tasks.
                    Will be emitted as 2-tuples `(LLM token, metadata)`.

                You can pass a list as the `stream_mode` parameter to stream multiple modes at once.
                The streamed outputs will be tuples of `(mode, data)`.
        """
        self.check_graph_available()

        graph: CompiledStateGraph = self._graph
        async for state in graph.astream(input_msg, config, stream_mode=stream_mode):
            yield state

    async def get_state_history(self,
                                config: RunnableConfig,
                                *,
                                use_filter: dict[str, Any] | None = None,
                                before: RunnableConfig | None = None,
                                limit: int | None = None) -> Sequence[StateSnapshot]:
        """Get the state history of the graph."""
        self.check_graph_available()

        graph: CompiledStateGraph = self._graph
        states: list[StateSnapshot] = []
        async for state in graph.aget_state_history(config, filter=use_filter, before=before, limit=limit):
            states.append(state)
        return states

    def get_state(self, config: RunnableConfig, *, sub_graphs: bool = False):
        self.check_graph_available()

        graph: CompiledStateGraph = self._graph
        return graph.get_state(config, subgraphs=sub_graphs)

    def delete_thread(self, thread_id: UUID):
        checkpointer = self._configurer.checkpointer
        if checkpointer is None:
            raise RuntimeError("Checkpointer is still not configured yet.")
        checkpointer.delete_thread(thread_id=str(thread_id))

    async def configure(self, force: bool = False):
        if self._is_configured and not force:
            self._logger.debug("Not forcefully configuring the agent. Skipping...")
            return
        self._logger.info("Configuring agent...")
        await self._configurer.async_configure()
        self._is_configured = True
        self._logger.info("Agent configured successfully!")

    async def restart(self):
        """
        Triggers the process of restarting the agent, updates its status, reconfigures,
        and rebuilds its internal graph. The function yields progress updates
        throughout the restart process.

        Returns:
            The progress of the restart operation.
        """
        statuses: list[Progress] = [
            {
                "status": "RESTARTING",
                "percentage": 0.0
            },
            {
                "status": "RESTARTING",
                "percentage": 0.6
            },
            {
                "status": "RESTARTED",
                "percentage": 1.0
            }
        ]
        self._logger.info("Restarting agent...")
        yield statuses[0]

        self._status = "RESTART"
        await self.configure(force=True)
        yield statuses[1]
        self.build_graph()
        self._status = "ON"
        yield statuses[2]

        self._logger.info("Agent restarted successfully!")

    async def embed_document(self, store_name: str, file_info: FileInformation):
        """
        Embeds a document into a specified vector store.

        This method takes a document's file information, loads it, splits it into
        chunks, and then embeds these chunks into the designated vector store.
        It generates unique UUIDs for each chunk to serve as their identifiers
        within the vector store.

        Args:
            store_name (str): The name of the vector store where the document
                chunks will be embedded. This name must correspond to a
                configured vector store.
            file_info (FileInformation): A dictionary containing information about
                the file to be embedded, including its 'path' and 'mime_type'.

        Raises:
            NotImplementedError: If the `store_name` provided does not correspond
                to any vector store configured in the system.
        """
        self._logger.debug("Embedding document...")

        vector_store = self._configurer.vector_store_configurer.get_store(store_name)
        if vector_store is not None:
            loader = get_document_loader(file_info["path"], file_info["mime_type"])
            splitter = self._configurer.text_splitter
            chunks = splitter.split_documents(loader.load())
            uuids = [str(uuid4()) for _ in range(len(chunks))]
            await vector_store.aadd_documents(documents=chunks, ids=uuids)

            self._logger.debug("Document embedded successfully!")
        else:
            raise NotImplementedError(f"No vector store {store_name} configured.")

    async def shutdown(self):
        self._logger.info("Shutting down Agent...")
        await self._configurer.async_destroy()
        self._logger.info("Good bye! See you later.")

    def build_graph(self):
        self._logger.info("Building graph...")

        self._logger.debug("Adding nodes to the graph...")
        graph = StateGraph(state_schema=State, config_schema=StateConfiguration, input=InputState)
        graph.add_node("query_or_respond", self._query_or_respond)
        graph.add_node("classify_data", self._classify_data)
        graph.add_node("suggest_questions", self._suggest_questions)

        tools = self._configurer.tools
        if tools is not None and len(tools) != 0:
            self._logger.debug("Adding tools to the graph...")
            graph.add_node("tools", ToolNode(tools))
            graph.add_conditional_edges("query_or_respond", tools_condition,
                                        {
                                            "tools": "tools",
                                            END: END,
                                        })
            graph.add_edge("tools", "query_or_respond")

        self._logger.debug("Adding edges to the graph...")
        graph.add_conditional_edges("classify_data", _routes_condition, {
            "query_or_respond": "query_or_respond",
            "suggest_questions": "suggest_questions"
        })
        graph.add_edge("suggest_questions", END)
        graph.set_entry_point("classify_data")

        self._logger.debug("Compiling the graph...")

        self._graph = graph.compile(name=self._configurer.config.agent_name,
                                    checkpointer=self._configurer.checkpointer)
        self._logger.info("Graph built successfully!")

    def check_graph_available(self):
        """
        Ensures that the agent's internal graph is properly initialized and that the agent
        is in an operational state before proceeding with any operations that require the graph.

        Raises:
            RuntimeError: If `self.is_configured` is `False`, indicating the agent has not been
                          configured. Users should call `configure()` to configure it.
            RuntimeError: If `self._graph` is `None`, indicating the agent graph has not been
                          initialized. Users should call `build_graph()` to initialize it.
            RuntimeError: If `self._status` is `"OFF"`, meaning the agent is currently turned off.
            RuntimeError: If `self._status` is `"RESTART"`, indicating the agent is in the process
                          of restarting.
        """
        if not self.is_configured:
            raise RuntimeError("The agent has not been configured yet. Please call `configure()` first.")
        if self._graph is None:
            raise RuntimeError("The agent graph has not been initialized yet. Please call `build_graph()` first.")
        if self._status == "OFF":
            raise RuntimeError("The agent is turned off.")
        if self._status == "RESTART":
            raise RuntimeError("The agent is restarting.")

    async def _query_or_respond(self, state: State) -> State:
        llm = self._configurer.llm
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._configurer.config.prompt.respond_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        messages = state["messages"]

        # Create prompt for the recognized topics
        classified_classes = state["classified_classes"]
        print(classified_classes[:2])

        if classified_classes is not None:
            topics = get_topics_from_classified_classes(classified_classes)
            topic_prompt = ("The provided questions may be related to the following topics:\n"
                            f'{'\n'.join(_convert_topics_to_str(topics))}') if len(topics) != 0 else None
            messages.append(SystemMessage(content=topic_prompt))

        # Create a real prompt to use
        prompt = prompt_template.invoke({
            "messages": messages
        })

        tools = self._configurer.tools
        chat_model = llm.bind_tools(tools=tools) if tools is not None else llm
        response = await chat_model.ainvoke(prompt)

        self._logger.debug(f"Response: {response}")

        return {
            "messages": [response],
            "classified_classes": state["classified_classes"]
        }

    async def _suggest_questions(self, state: State) -> State:
        classified_classes = state["classified_classes"]
        topics = get_topics_from_classified_classes(classified_classes)
        prompt = PromptTemplate.from_template(self._configurer.config.respond_prompt.suggest_questions_prompt)
        response = await self._configurer.llm.ainvoke(
            prompt.format(topics="\n".join(_convert_topics_to_str(topics))))

        self._logger.debug(f"Response: {response}")

        return {
            "messages": [response],
            "classified_classes": state["classified_classes"]
        }

    async def _classify_data(self, state: InputState) -> State:
        messages = state["messages"]
        image_recognizer = self._configurer.image_recognizer
        image_paths = state["image_paths"]
        if image_recognizer is None or len(image_paths) == 0:
            return {"classified_classes": None, "messages": messages}
        if not image_recognizer.is_initialized:
            self._logger.warning('Image recognizer has not been initialized yet.')
            return {"classified_classes": None, "messages": messages}

        async def recognize_image(image_path: str):
            prediction_result = await image_recognizer.async_predict(image_path)

            results: list[ClassifiedClass] = []
            for i in range(len(prediction_result["classes"])):
                class_name = prediction_result["classes"][i]
                prob = prediction_result["probabilities"][i]
                results.append({
                    "class_name": class_name,
                    "probability": prob,
                    "data_type": "image",
                })
            return results

        recognize_image_tasks = [recognize_image(img_path) for img_path in image_paths]
        try:
            topics: list[ClassifiedClass] = await asyncio.gather(*recognize_image_tasks)
        except Exception as e:
            self._logger.warning(f"\nCaught an unexpected error: {type(e).__name__}: {e}")
            topics = []

        return {"classified_classes": topics, "messages": messages}

    @property
    def configurer(self):
        return self._configurer

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value: Literal["ON", "OFF"]):
        self._status = value

    @property
    def is_configured(self):
        return self._is_configured

    @property
    def graph(self):
        return self._graph
