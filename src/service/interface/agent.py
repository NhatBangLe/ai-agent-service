import datetime
from abc import ABC, abstractmethod
from typing import Literal, Any, TypedDict, Sequence
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.types import StateSnapshot
from pydantic import BaseModel, Field

from src.config.configurer.interface.agent import AgentConfigurer
from src.util import FileInformation


class Attachment(BaseModel):
    id: str = Field(description="Unique identifier of the attachment.", min_length=1)
    name: str = Field(description="Name of the attachment.", min_length=1)
    mime_type: str = Field(description="MIME type of the attachment.", min_length=1)
    path: str = Field(description="Path to the attachment.")


class StateConfiguration(TypedDict):
    pass


class State(MessagesState):
    pass


class AgentMetadata(BaseModel):
    status: Literal["ON", "OFF", "RESTART", "EMBED_DOCUMENT"] = Field(description="Current Agent status.")
    available_vector_stores: Sequence[str] = Field(description="A sequence contains names of available vector stores.")
    bm25_last_sync: datetime.datetime | None = Field(description="The last sync time of BM25 retriever.")


class IAgentService(ABC):

    @abstractmethod
    def stream(self, input_state: State, config: RunnableConfig | None = None, *,
               stream_mode: Literal["values", "updates", "messages"] | None = None):
        """
        Streams states from the compiled state graph based on the input state, configuration,
        and stream mode, yielding updated states for further processing.

        Args:
            input_state: The input to the graph.
            config: The configuration to use for the run.
            stream_mode: The mode to stream output, defaults to `self.stream_mode`. Options are:

                - `"values"`: Emit all values in the state after each step, including interrupts.
                    When used with functional API, values are emitted once at the end of the workflow.
                - `"updates"`: Emit only the node or task names and updates returned by the nodes or tasks after each step.
                    If multiple updates are made in the same step (e.g., multiple nodes are run), then those updates are emitted separately.
                - `"messages"`: Emit LLM messages token-by-token together with metadata for any LLM invocations inside nodes or tasks.
                    Will be emitted as 2-tuples `(LLM token, metadata)`.

        Returns:
            A generator yielding updated states of type State.
        """
        pass

    @abstractmethod
    async def astream(self, input_state: State, config: RunnableConfig | None = None, *,
                      stream_mode: Literal["values", "updates", "messages"] | None = None):
        pass

    @abstractmethod
    async def get_state_history(self,
                                config: RunnableConfig,
                                *,
                                use_filter: dict[str, Any] | None = None,
                                before: RunnableConfig | None = None,
                                limit: int | None = None) -> list[StateSnapshot]:
        """
        Retrieve the state history for the current graph based on the provided
        parameters. This method operates asynchronously and returns a sequence
        of state snapshots, filtering the results if specified. The method will
        check the availability of the graph before attempting to fetch the state
        history.

        :param config: The configuration for which the state history is being
            fetched. It is used to define the specific execution context.
        :param use_filter: A dictionary specifying filter conditions to refine
            the state history results. If None, no filters will be applied.
        :param before: Specifies a point in the state timeline to retrieve
            historical states prior to it. If None, all states will be retrieved.
        :param limit: An integer defining the maximum number of states to be
            returned. If None, all available states up to the specified
            conditions will be fetched.
        :return: A sequence containing the state snapshots corresponding to the
            specified configuration and filter criteria.
        """
        pass

    @abstractmethod
    def get_state(self, config: RunnableConfig, *, sub_graphs: bool = False):
        pass

    @abstractmethod
    async def delete_all_checkpoints_by_thread_id(self, thread_id: UUID):
        pass

    @abstractmethod
    async def configure(self, force: bool = False):
        pass

    @abstractmethod
    async def restart(self):
        pass

    @abstractmethod
    async def embed_document(self, store_name: str, file_info: FileInformation):
        pass

    @abstractmethod
    async def unembed_document(self, store_name: str, chunk_ids: list[str]):
        pass

    @abstractmethod
    async def shutdown(self):
        pass

    @abstractmethod
    def build_graph(self):
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def set_status(self, value: Literal["ON", "OFF"]):
        pass

    @property
    @abstractmethod
    def configurer(self) -> AgentConfigurer:
        pass

    @property
    @abstractmethod
    def metadata(self) -> AgentMetadata:
        pass

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        pass
