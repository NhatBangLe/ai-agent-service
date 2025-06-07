import asyncio
import logging
from logging import Logger
from typing import Literal, Any
from uuid import uuid4, UUID

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.state import State, InputState, Configuration, Attachment, ClassifiedClass
from src.config.configurer.agent import AgentConfigurer
from src.data.database import create_session
from src.data.model import Image
from src.util.main import FileInformation, Progress


def _routes_condition(state: State) -> Literal["suggest_questions", "query_or_respond"]:
    latest_messages = state["messages"][-1]
    classified_classes = state["classified_classes"]

    if not isinstance(latest_messages, HumanMessage) and classified_classes is not None and len(
            classified_classes) != 0:
        return "suggest_questions"
    return "query_or_respond"


def get_document_loader(file_path: str | bytes, mime_type: str) -> BaseLoader:
    if mime_type == "application/pdf":
        return PyPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported MIME type: {mime_type}")


class Agent:
    _status: Literal["ON", "OFF", "RESTART"]
    _configurer: AgentConfigurer
    _graph: CompiledStateGraph | None
    _checkpointer: BaseCheckpointSaver[Any] | None
    _is_configured: bool
    _logger: Logger

    def __init__(self, configurer: AgentConfigurer):
        self._status = "ON"
        self._configurer = configurer
        self._graph = None
        self._is_configured = False
        self._logger = logging.getLogger(__name__)

    def stream(self, input_msg: InputState, config: RunnableConfig | None = None,
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
        if self._graph is None:
            raise RuntimeError("The agent graph has not been initialized yet. Please call `build_graph()` first.")
        graph: CompiledStateGraph = self._graph
        for state in graph.stream(input_msg, config, stream_mode=stream_mode):
            yield state

    # noinspection PyShadowingBuiltins
    def get_state_history(self,
                          config: RunnableConfig,
                          *,
                          filter: dict[str, Any] | None = None,
                          before: RunnableConfig | None = None,
                          limit: int | None = None):
        """Get the state history of the graph."""
        if self._graph is None:
            raise RuntimeError("Graph is still not compiled yet.")
        graph: CompiledStateGraph = self._graph
        return graph.get_state_history(config, filter=filter, before=before, limit=limit)

    def get_state(self, config: RunnableConfig, *, sub_graphs: bool = False):
        if self._graph is None:
            raise RuntimeError("Graph is still not compiled yet.")
        graph: CompiledStateGraph = self._graph
        return graph.get_state(config, subgraphs=sub_graphs)

    def configure(self, force: bool = False):
        if self._is_configured and not force:
            self._logger.debug("Not forcefully configuring the agent. Skipping...")
            return
        self._logger.info("Configuring agent...")
        self._configurer.configure()
        self._is_configured = True
        self._logger.info("Agent configured successfully!")

    def restart(self):
        """
        Triggers the process of restarting the agent, updates its status, reconfigures,
        and rebuilds its internal graph. The function yields progress updates
        throughout the restart process.

        Returns:
            A string representing the progress of the restart operation.
            `{"status": "RESTARTING", "percentage": 0.0}`, use a new line character to separate lines.
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
        yield str(f'{statuses[0]}\n')

        self._status = "RESTART"
        self._configurer.configure()
        yield str(f'{statuses[1]}\n')
        self.build_graph()
        self._status = "ON"
        yield str(f'{statuses[2]}\n')

        self._logger.info("Agent restarted successfully!")

    def embed_document(self, file_info: FileInformation):
        self._logger.debug("Embedding document...")

        vector_store = self._configurer.vector_store
        if vector_store is not None:
            loader = get_document_loader(file_info["path"], file_info["mime_type"])
            splitter = self._configurer.get_text_splitter()
            chunks = splitter.split_documents(loader.load())
            uuids = [str(uuid4()) for _ in range(len(chunks))]
            vector_store.add_documents(documents=chunks, ids=uuids)

            self._logger.debug("Document embedded successfully!")
        else:
            self._logger.debug("No vector store configured. Skipping embedding.")

    def build_graph(self):
        self._logger.info("Building graph...")

        self._logger.debug("Adding nodes to the graph...")
        graph = StateGraph(state_schema=State, config_schema=Configuration, input=InputState)
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

        from ..data.database import url
        conn_str = f"postgresql://{url.username}:{url.password}@{url.host}:{url.port}/{url.database}"
        with PostgresSaver.from_conn_string(conn_str) as checkpointer:
            checkpointer.setup()
            self._checkpointer = checkpointer

        self._graph = graph.compile(name=self._configurer.config.agent_name,
                                    checkpointer=self._checkpointer)
        self._logger.info("Graph built successfully!")

    async def _query_or_respond(self, state: State) -> State:
        llm = self._configurer.llm
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._configurer.config.prompt.respond_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])

        # Create prompt for the recognized topics
        classified_classes = state["classified_classes"]
        topic_prompt: str | None = None
        if classified_classes is not None:
            topics = self.get_topics_from_classified_classes(classified_classes)
            topic_prompt = ("The provided questions may be related to the following topics:"
                            f'{'\n'.join(topics)}') if len(topics) != 0 else None

        # Create a real prompt to use
        messages = (state["messages"] + [SystemMessage(content=topic_prompt)]
                    if topic_prompt is not None else state["messages"])
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
        topics: list[str] = self.get_topics_from_classified_classes(classified_classes)

        prompt = PromptTemplate.from_template(self._configurer.config.respond_prompt.suggest_questions_prompt)
        response = await self._configurer.llm.ainvoke(prompt.format(topics="\n".join(topics)))

        self._logger.debug(f"Response: {response}")

        return {
            "messages": [response],
            "classified_classes": state["classified_classes"]
        }

    async def _classify_data(self, state: InputState) -> State:
        messages = state["messages"]
        image_recognizer = self._configurer.image_recognizer
        if image_recognizer is None:
            return {"classified_classes": None, "messages": messages}

        attachments = state["attachments"]
        images: list[Attachment] = [att for att in attachments if att["mime_type"].__contains__("image")]

        async def recognize_image(image_id: UUID):
            with create_session() as session:
                db_image = session.get(Image, image_id)
                image_path = db_image.save_path
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

        recognize_image_tasks = [recognize_image(img["image_id"]) for img in images]
        try:
            topics: list[ClassifiedClass] = await asyncio.gather(*recognize_image_tasks)
        except Exception as e:
            self._logger.warning(f"\nCaught an unexpected error: {type(e).__name__}: {e}")
            topics = []

        return {"classified_classes": topics, "messages": messages}

    def get_topics_from_classified_classes(self, classified_classes: list[ClassifiedClass]):
        image_class_descriptors = self._configurer.config.image_recognizer.classes

        topics: list[str] = []
        for classified_class in classified_classes:
            if classified_class["data_type"] == "image":
                found_topics = [descriptor.description for descriptor in image_class_descriptors
                                if descriptor.name == classified_class["class_name"]]
                topics.append(*found_topics)
            else:
                raise NotImplementedError(f'Classified class for text data has not been supported yet.')
        return topics

    @property
    def configurer(self):
        return self._configurer

    @property
    def status(self):
        return self._status

    @property
    def is_configured(self):
        return self._is_configured
