import asyncio
import logging
from logging import Logger
from typing import Literal
from uuid import uuid4

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_experimental.text_splitter import SemanticChunker
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import StateSnapshot, RetryPolicy

from src.config.configurer.interface.agent import AgentConfigurer
from src.service.interface.agent import Attachment, StateConfiguration, State, AgentMetadata, IAgentService
from src.util import FileInformation, Progress
from src.util.constant import SUPPORTED_LANGUAGE_DICT


class Agent(IAgentService):
    _status: Literal["ON", "OFF", "RESTART", "EMBED_DOCUMENT"]
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

    def stream(self, input_state, config=None, *, stream_mode=None):

        self.check_graph_available()

        graph: CompiledStateGraph = self._graph
        for state in graph.stream(input_state, config, stream_mode=stream_mode):
            yield state

    async def astream(self, input_state, config=None, *, stream_mode=None):
        """
        Args:
            input_state: The input to the graph.
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
        async for state in graph.astream(input_state, config, stream_mode=stream_mode):
            yield state

    async def get_state_history(self, config, *, use_filter=None, before=None, limit=None):
        self.check_graph_available()

        graph: CompiledStateGraph = self._graph
        states: list[StateSnapshot] = []
        async for state in graph.aget_state_history(config, filter=use_filter, before=before, limit=limit):
            states.append(state)
        return states

    def get_state(self, config, *, sub_graphs=False):
        self.check_graph_available()

        graph: CompiledStateGraph = self._graph
        return graph.get_state(config, subgraphs=sub_graphs)

    def delete_all_checkpoints_by_thread_id(self, thread_id):
        checkpointer = self._configurer.checkpointer
        if checkpointer is None:
            raise RuntimeError("Checkpointer is still not configured yet.")
        checkpointer.delete_thread(thread_id=str(thread_id))

    async def configure(self, force=False):
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
                "status": "RESTART",
                "percentage": 0.0
            },
            {
                "status": "RESTART",
                "percentage": 0.6
            },
            {
                "status": "ON",
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

    async def embed_document(self, store_name, file_info):
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
            ValueError: If the `store_name` provided does not correspond
                to any vector store configured in the system.
        """
        self._status = "EMBED_DOCUMENT"
        self._logger.debug("Embedding document...")

        vector_store = self._configurer.vector_store_configurer.get_store(store_name)
        if vector_store is not None:
            self._logger.debug(f'Retrieving content from {file_info["name"]} document...')

            # Convert a file to a Document object
            doc_path = file_info["path"]
            converter = DocumentConverter()
            result = converter.convert(doc_path)
            document = Document(page_content=result.document.export_to_markdown(),
                                metadata={"source": doc_path, "total_pages": len(result.pages)})

            chunker = SemanticChunker(vector_store.embeddings)
            self._logger.debug(f'Splitting documents by using semantic similarity...')

            def split_docs():
                return chunker.split_documents([document])

            chunks = await asyncio.to_thread(split_docs)

            self._logger.debug(f'Adding chunks to vector store {store_name}...')
            uuids = [str(uuid4()) for _ in range(len(chunks))]
            added_ids = await vector_store.aadd_documents(documents=chunks, ids=uuids)

            self._logger.debug("Document embedded successfully!")
        else:
            raise ValueError(f"No vector store {store_name} configured.")

        self._status = "ON"
        return added_ids

    async def unembed_document(self, store_name, chunk_ids):
        self._logger.debug("Unembedding documents...")

        vector_store = self._configurer.vector_store_configurer.get_store(store_name)
        if vector_store is not None:
            await vector_store.adelete(ids=chunk_ids)
            self._logger.debug("Documents have been unembedded successfully!")
        else:
            raise ValueError(f"No vector store {store_name} configured.")

    async def shutdown(self):
        self._logger.info("Shutting down Agent...")
        await self._configurer.async_destroy()
        self._logger.info("Good bye! See you later.")

    def build_graph(self):
        self._logger.info("Building graph...")
        graph = StateGraph(state_schema=State, config_schema=StateConfiguration)

        # Add nodes
        self._logger.debug("Adding nodes to the graph...")
        graph.add_node("query_or_respond", self._query_or_respond, retry=RetryPolicy())
        graph.add_node("generate_answer", self._generate_answer, retry=RetryPolicy())
        graph.add_node("tools", ToolNode(self._configurer.tools))

        # Add edges
        self._logger.debug("Add edges to the graph...")
        graph.add_conditional_edges("query_or_respond", tools_condition,
                                    {"tools": "tools", END: END})
        graph.add_edge("tools", "generate_answer")
        graph.add_edge(START, "query_or_respond")
        graph.add_edge("generate_answer", END)

        # Compile graph
        self._logger.debug("Compiling the graph...")
        self._graph = graph.compile(name=self._configurer.config.agent_name,
                                    checkpointer=self._configurer.checkpointer)
        self._logger.info("Graph built successfully!")

    def check_graph_available(self):
        if not self.is_configured:
            raise RuntimeError("The agent has not been configured yet. Please call `configure()` first.")
        if self._graph is None:
            raise RuntimeError("The agent graph has not been initialized yet. Please call `build_graph()` first.")
        if self._status == "OFF":
            raise RuntimeError("The agent is turned off.")
        if self._status == "RESTART":
            raise RuntimeError("The agent is restarting.")

    async def _query_or_respond(self, state: State, config: RunnableConfig):
        lang = self._configurer.config.language
        messages = state["messages"]
        attachment: Attachment | None = messages[-1].additional_kwargs["attachment"]
        system_msgs: list[SystemMessage] = [
            SystemMessage(content=f'Your primary language is {SUPPORTED_LANGUAGE_DICT[lang]}.'),
        ]

        if attachment is not None:
            prompt_template = ChatPromptTemplate.from_messages([
                *system_msgs,
                SystemMessage(
                    content="If you are given an attachment. You should analysis based on the following steps:\n"
                            "*Step 1. Specify what attachment type you have by looking at a MIME type of the attachment.\n"
                            "Step 2. Select the correct recognition tool to use based on the attachment type.\n"
                            "Step 3. Call the selected recognition tool.*\n"
                            "**IMPORTANT NOTICE:**"
                            "1. If you don't have compatible tools, you MUST say that you don't have "
                            "compatible tools to recognize the attachment.\n"
                            "2. Remember to tell us what you are doing.\n"
                            "3. You MUST NOT say the details of the provided attachment."),
                MessagesPlaceholder(variable_name="messages"),
                HumanMessage(content=f"You are given an attachment. There is the attachment information:\n"
                                     f"{attachment.model_dump_json()}")
            ])
            prompt = await prompt_template.ainvoke({"messages": messages[:-1]}, config)
        else:
            prompt_template = ChatPromptTemplate.from_messages([*system_msgs,
                                                                MessagesPlaceholder(variable_name="messages")])
            prompt = await prompt_template.ainvoke({"messages": messages}, config)

        self._logger.debug(f'Constructed prompt:\n{prompt}\n')

        response = await self._configurer.chat_model.ainvoke(prompt)
        self._logger.debug(f"Response: {response}")

        return {"messages": [response]}

    async def _generate_answer(self, state: State, config: RunnableConfig):
        lang = self._configurer.config.language

        # Create a prompt
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=f'Your primary language is {SUPPORTED_LANGUAGE_DICT[lang]}.'),
            SystemMessage(content=self._configurer.config.prompt.respond_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        prompt = await prompt_template.ainvoke({"messages": state["messages"]}, config)
        self._logger.debug(f'Constructed prompt:\n{prompt}\n')

        response = await self._configurer.chat_model.ainvoke(prompt, config)
        self._logger.debug(f"Response: {response}")

        return {"messages": [response]}

    def set_status(self, value: Literal["ON", "OFF"]):
        self._status = value

    @property
    def configurer(self):
        return self._configurer

    @property
    def metadata(self):
        bm25_configurer = self._configurer.bm25_configurer
        bm25_last_sync = bm25_configurer.last_sync if bm25_configurer is not None else None

        vs_configurer = self._configurer.vector_store_configurer
        if vs_configurer is None:
            available_vector_stores = []
        else:
            all_store_configs = vs_configurer.get_all_configs()
            available_vector_stores = [config.name for config in all_store_configs]

        return AgentMetadata(name=self._configurer.config.agent_name,
                             description=self._configurer.config.description,
                             status=self._status,
                             bm25_last_sync=bm25_last_sync,
                             available_vector_stores=available_vector_stores)

    @property
    def is_configured(self):
        return self._is_configured

    @property
    def graph(self):
        return self._graph
