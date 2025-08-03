import asyncio
from logging import Logger, getLogger
from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from src.config.configurer.interface.chat_model import ChatModelConfigurer
from src.config.model.chat_model import ChatModelConfiguration, ChatModelType
from src.config.model.chat_model.google_genai import convert_safety_settings_to_genai, GoogleGenAIChatModelConfiguration
from src.config.model.chat_model.ollama import OllamaChatModelConfiguration


class ChatModelConfigurerImpl(ChatModelConfigurer):
    _models: dict[str, tuple[ChatModelConfiguration, BaseChatModel]]
    _logger: Logger = getLogger(__name__)

    def __init__(self):
        super().__init__()
        self._models = {}

    def configure(self, config, /, **kwargs):
        self._logger.debug("Configuring chat model...")

        if config.type == ChatModelType.GOOGLE_GENAI:
            genai = cast(GoogleGenAIChatModelConfiguration, config)
            safety_settings = convert_safety_settings_to_genai(genai.safety_settings) if genai.safety_settings else None
            llm = init_chat_model(
                model_provider="google_genai",
                model=genai.model_name,
                temperature=genai.temperature,
                timeout=genai.timeout,
                max_tokens=genai.max_tokens,
                max_retries=genai.max_retries,
                top_p=genai.top_p,
                top_k=genai.top_k,
                safety_settings=safety_settings)
        elif config.type == ChatModelType.OLLAMA:
            ollama = cast(OllamaChatModelConfiguration, config)
            llm = init_chat_model(
                model_provider="ollama",
                model=ollama.model_name,
                base_url=ollama.base_url,
                temperature=ollama.temperature,
                seed=ollama.seed,
                num_ctx=ollama.num_ctx,
                num_predict=ollama.num_predict,
                repeat_penalty=ollama.repeat_penalty,
                stop=ollama.stop,
                top_p=ollama.top_p,
                top_k=ollama.top_k)
        else:
            raise NotImplementedError(f'{config} is not supported.')

        self._models[config.model_name] = (config, llm)
        self._logger.debug(f"Chat model {config.model_name} has been configured successfully.")

    async def async_configure(self, config, /, **kwargs):
        await asyncio.to_thread(self.configure, config, **kwargs)

    def destroy(self, **kwargs):
        pass

    async def async_destroy(self, **kwargs):
        pass

    def get_models(self):
        if self._models is None:
            return []
        return [tool for _, (_, tool) in self._models.items()]

    def get_model(self, unique_name: str):
        if self._models is None:
            return None
        value = self._models[unique_name]
        return value[1] if value is not None else None

    def get_config(self, unique_name: str):
        if self._models is None:
            return None
        value = self._models[unique_name]
        return value[0] if value is not None else None
