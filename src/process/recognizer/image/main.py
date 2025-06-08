import logging
import os.path
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from typing import Any

import jsonpickle

from src.config.model.recognizer.image.main import ImageRecognizerConfiguration
from src.process.recognizer.main import Recognizer, RecognizerOutput
from src.util.function import get_config_folder_path


class ImageRecognizer(Recognizer):
    """
    PyTorch inference class using TorchScript
    """
    _config: ImageRecognizerConfiguration
    num_classes: int | None
    is_initialized: bool
    _executor: ThreadPoolExecutor
    _logger: Logger

    def __init__(self,
                 config: ImageRecognizerConfiguration,
                 max_workers: int = 4,
                 **data: Any):
        """
        Initialize the inference engine

        Args:
            config: An object configuration of image recognizer.
            max_workers: Number of worker threads for async processing, min value 1
        """
        super().__init__(**data)
        self._config = config
        self.num_classes = None
        self.is_initialized = False
        self._executor = ThreadPoolExecutor(max_workers=max(max_workers, 1))

        # Setup logging
        self._logger = logging.getLogger(__name__)

    def configure(self):
        # Load and optimize model

        # Load output classes
        path = os.path.join(get_config_folder_path(), self._config.output_config_path)
        with open(path, "r") as config_file:
            json = config_file.read()
        output = RecognizerOutput.model_validate(jsonpickle.decode(json))
        self.num_classes = len(output.classes)

        self.is_initialized = True
        self._logger.info(f"Number of classes: {self.num_classes}.")
