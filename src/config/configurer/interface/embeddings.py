from abc import abstractmethod
from typing import Sequence

from langchain_core.embeddings import Embeddings

from . import Configurer
from ...model.embeddings import EmbeddingsConfiguration


class EmbeddingsConfigurer(Configurer):

    @abstractmethod
    def configure(self, config: EmbeddingsConfiguration, /, **kwargs):
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_model(self, unique_name: str) -> Embeddings | None:
        pass

    @abstractmethod
    def get_model_config(self, unique_name: str) -> EmbeddingsConfiguration | None:
        pass

    @abstractmethod
    def get_all_stores(self) -> Sequence[Embeddings]:
        pass
