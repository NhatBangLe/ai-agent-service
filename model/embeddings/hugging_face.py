from model.embeddings.main import EmbeddingsModelConfiguration, EmbeddingsModelProvider


class HuggingFaceEmbeddingsConfiguration(EmbeddingsModelConfiguration):
    """
    Embeddings model class for the embeddings_model property in configuration files.
    """
    provider = EmbeddingsModelProvider.HUGGING_FACE
