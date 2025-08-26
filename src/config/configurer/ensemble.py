import datetime

from langchain.retrievers import EnsembleRetriever
from langchain_core.tools import create_retriever_tool

from src.config.configurer.interface.ensemble import EnsembleRetrieverConfigurer
from src.util.function import get_datetime_now


class EnsembleRetrieverConfigurerImpl(EnsembleRetrieverConfigurer):
    _retriever: EnsembleRetriever | None = None
    _last_modified: datetime.datetime | None = None

    RETRIEVER_NAME = "ensemble_retriever"
    RETRIEVER_DESCRIPTION = (
        "A highly robust and comprehensive tool designed to retrieve the most relevant and "
        "accurate information from a vast knowledge base by combining multiple advanced search algorithms."
        "**USE THIS TOOL WHENEVER THE USER ASKS A QUESTION REQUIRING EXTERNAL KNOWLEDGE,"
        "FACTUAL INFORMATION, CURRENT EVENTS, OR DATA BEYOND YOUR INTERNAL TRAINING.**"
        "**Examples of when to use this tool:**"
        "- \"the capital of France?\""
        "- \"the history of the internet.\""
        "- \"the latest developments in AI?\""
        "- \"quantum entanglement.\""
        "**Crucially, use this tool for any query that cannot be answered directly from your"
        "pre-trained knowledge, especially if it requires up-to-date, specific, or detailed factual data.**"
        "The tool takes a single, concise search query as input."
        "If you cannot answer after using this tool, you can use another tool to retrieve more information.")

    def configure(self, retrievers, weights, **kwargs):
        if retrievers is None or len(retrievers) == 0 or weights is None or len(weights) == 0:
            return

        self._retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)
        self._last_modified = get_datetime_now()

    async def async_configure(self, retrievers, weights, **kwargs):
        self.configure(retrievers, weights, **kwargs)

    def destroy(self, **kwargs):
        pass

    async def async_destroy(self, **kwargs):
        self.destroy()

    @property
    def tool(self):
        if self._retriever is None:
            return None
        return create_retriever_tool(
            self._retriever,
            name=self.RETRIEVER_NAME,
            description=self.RETRIEVER_DESCRIPTION)

    @property
    def retriever(self):
        return self._retriever
