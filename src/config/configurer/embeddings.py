import asyncio
import logging

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from .interface.embeddings import EmbeddingsConfigurer
from ..model.embeddings import EmbeddingsConfiguration
from ..model.embeddings.google_genai import GoogleGenAIEmbeddingsConfiguration
from ..model.embeddings.hugging_face import HuggingFaceEmbeddingsConfiguration


class EmbeddingsConfigurerImpl(EmbeddingsConfigurer):
    _embeddings: dict[str, tuple[EmbeddingsConfiguration, Embeddings]] | None = None
    _logger = logging.getLogger(__name__)

    def configure(self, config, /, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_configure(config, **kwargs))

    async def async_configure(self, config, /, **kwargs):
        self._logger.debug(f"Configuring embedding model {config.name}...")
        if self._embeddings is None:
            self._embeddings = {}

        if isinstance(config, HuggingFaceEmbeddingsConfiguration):
            model = HuggingFaceEmbeddings(model_name=config.model_name)
        elif isinstance(config, GoogleGenAIEmbeddingsConfiguration):
            task_type = str(config.task_type.value) if config.task_type is not None else None
            model = GoogleGenerativeAIEmbeddings(
                model=config.model_name,
                task_type=task_type,
            )
        else:
            raise NotImplementedError(f'{type(config)} is not supported.')

        self._embeddings[config.name] = (config, model)
        self._logger.debug("Configured embeddings model successfully.")

    def destroy(self, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_destroy(**kwargs))

    async def async_destroy(self, **kwargs):
        pass

    def get_model(self, unique_name):
        if self._embeddings is None:
            self._logger.debug("No models have been configured yet.")
            return None
        value = self._embeddings[unique_name]
        return value[1] if value is not None else None

    def get_model_config(self, unique_name):
        if self._embeddings is None:
            self._logger.debug("No stores has been configured yet.")
            return None
        value = self._embeddings[unique_name]
        return value[0] if value is not None else None

    def get_all_stores(self):
        if self._embeddings is None:
            return []
        return [store for _, (_, store) in self._embeddings.items()]
