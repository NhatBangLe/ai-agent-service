import asyncio
import logging
import os

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import ToolException, BaseTool, ArgsSchema
from pydantic import BaseModel, Field

from src.config.configurer import Configurer
from src.config.model.recognizer.image import ImageRecognizerConfiguration
from src.process.recognizer.image import ImageRecognizer
from src.util.function import get_topics_from_class_names


class RecognizeImageInput(BaseModel):
    image_path: str = Field(description="Path to the image file.")


class RecognizeImageTool(BaseTool):
    name: str = "image_recognizer"
    description: str = ("Useful for recognizing an image. "
                        "Returns recognized topics and their probabilities for the given image file.")
    args_schema: ArgsSchema | None = RecognizeImageInput
    image_recognizer: ImageRecognizer | None = None

    def _run(self, *args, **kwargs) -> str:
        raise ToolException("Image recognizer is not available in synchronous mode.")

    async def _arun(self, image_path: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        try:
            prediction_result = await self.image_recognizer.async_predict(image_path)
        except RuntimeError as e:
            raise ToolException(e)

        topics = get_topics_from_class_names(prediction_result["classes"])
        holder: dict[str, tuple[str, float]] = {}
        for name, prob in zip(prediction_result["classes"], prediction_result["probabilities"]):
            holder[name] = (topics[name], prob)
        result = (f'Result from recognizing image: '
                  f'{"; ".join([f'Topic: {topic} - Accuracy: {round(prob * 100, 2)}%'
                                for topic, prob in holder.values()])}')
        run_manager.on_tool_end(result)

        return result


class ImageRecognizerConfigurer(Configurer):
    _config: ImageRecognizerConfiguration | None = None
    _image_recognizer: ImageRecognizer | None = None
    _tool: RecognizeImageTool | None = None
    _logger = logging.getLogger(__name__)

    def configure(self, config: ImageRecognizerConfiguration, **kwargs):
        self._logger.debug("Configuring image recognizer...")
        self._config = config

        if config is None or config.enable is False:
            self._logger.info("Image recognizer is disabled.")
        else:
            max_workers = os.getenv("IMAGE_RECOGNIZER_MAX_WORKERS", "4")
            recognizer = ImageRecognizer(config=config, max_workers=int(max_workers))
            recognizer.configure()

            self._logger.debug("Configured image recognizer successfully.")
            self._image_recognizer = recognizer
            self._tool = RecognizeImageTool(image_recognizer=recognizer)

    async def async_configure(self, config: ImageRecognizerConfiguration, **kwargs):
        self.configure(config, **kwargs)

    def destroy(self, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_destroy(**kwargs))

    async def async_destroy(self, **kwargs):
        pass

    @property
    def tool(self):
        return self._tool

    @property
    def image_recognizer(self):
        return self._image_recognizer
