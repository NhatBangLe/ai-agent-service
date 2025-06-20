import asyncio
import logging
from typing import Sequence

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.config.configurer import Configurer
from src.config.model.embeddings import EmbeddingsConfiguration
from src.config.model.embeddings.hugging_face import HuggingFaceEmbeddingsConfiguration


class EmbeddingsConfigurer(Configurer):
    _embeddings: dict[str, tuple[EmbeddingsConfiguration, Embeddings]] | None = None
    _logger = logging.getLogger(__name__)

    def configure(self, config: EmbeddingsConfiguration, /, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_configure(config, **kwargs))

    async def async_configure(self, config: EmbeddingsConfiguration, /, **kwargs):
        """
        Configures the embedding model for text embedding generation.

        This method initializes the `self._embeddings_model` attribute based on
        the provided `EmbeddingsModelConfiguration`.

        Args:
            config: An instance of `EmbeddingsModelConfiguration` containing the
                configuration parameters for the embedding model.

        Raises:
            TypeError: If the embedding model provider specified in the
                configuration is not currently supported.
        """
        self._logger.debug(f"Configuring embedding model {config.name}...")
        if self._embeddings is None:
            self._embeddings = {}

        if isinstance(config, HuggingFaceEmbeddingsConfiguration):
            model = HuggingFaceEmbeddings(model_name=config.model_name)
        else:
            raise NotImplementedError(f'{type(config)} is not supported.')

        self._embeddings[config.name] = (config, model)
        self._logger.debug("Configured embeddings model successfully.")

    def destroy(self, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_destroy(**kwargs))

    async def async_destroy(self, **kwargs):
        pass

    def get_model(self, unique_name: str) -> Embeddings | None:
        if self._embeddings is None:
            self._logger.debug("No models have been configured yet.")
            return None
        value = self._embeddings[unique_name]
        return value[1] if value is not None else None

    def get_model_config(self, unique_name: str) -> EmbeddingsConfiguration | None:
        if self._embeddings is None:
            self._logger.debug("No stores has been configured yet.")
            return None
        value = self._embeddings[unique_name]
        return value[0] if value is not None else None

    def get_all_stores(self) -> Sequence[Embeddings]:
        if self._embeddings is None:
            return []
        return [store for _, (_, store) in self._embeddings.items()]
