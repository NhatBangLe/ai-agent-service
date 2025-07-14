import asyncio
import logging
import os
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from logging import Logger
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import requests
import torch
from PIL import Image
from torch import jit as jit
from torchvision.transforms import Resize, Normalize, CenterCrop, Pad, Grayscale, Compose, ToTensor

from src.config.model.recognizer.image import ImagePreprocessingConfiguration, ImageRecognizerConfiguration
from src.config.model.recognizer.image.preprocessing import ImageResizeConfiguration, ImageNormalizeConfiguration, \
    ImageCenterCropConfiguration, ImagePadConfiguration, ImageGrayscaleConfiguration, INTERPOLATION_MODE_DICT
from src.process.recognizer import Recognizer, RecognizingResult, RecognizerOutput
from src.util.function import get_config_folder_path, is_web_path

__all__ = ["get_transform_layer", "ImageRecognizer"]


def get_transform_layer(config: ImagePreprocessingConfiguration) -> torch.nn.Module:
    if isinstance(config, ImageResizeConfiguration):
        resize = typing.cast(ImageResizeConfiguration, config)

        return Resize(
            size=(resize.target_size,),
            interpolation=INTERPOLATION_MODE_DICT[resize.interpolation],
            max_size=resize.max_size,
            antialias=resize.antialias
        )
    elif isinstance(config, ImageNormalizeConfiguration):
        normalize = typing.cast(ImageNormalizeConfiguration, config)

        return Normalize(
            mean=normalize.mean,
            std=normalize.std,
            inplace=normalize.inplace
        )
    elif isinstance(config, ImageCenterCropConfiguration):
        center_crop = typing.cast(ImageCenterCropConfiguration, config)

        return CenterCrop(size=center_crop.size)
    elif isinstance(config, ImagePadConfiguration):
        pad = typing.cast(ImagePadConfiguration, config)

        return Pad(
            padding=pad.padding,
            fill=pad.fill,
            padding_mode=pad.padding_mode
        )
    elif isinstance(config, ImageGrayscaleConfiguration):
        grayscale = typing.cast(ImageGrayscaleConfiguration, config)

        return Grayscale(num_output_channels=grayscale.num_output_channels)
    else:
        raise NotImplementedError(f'Image preprocessing layer: {type(config)} is not supported.')


class ImageRecognizer(Recognizer):
    """
    PyTorch inference class using TorchScript
    """
    _config: ImageRecognizerConfiguration
    _device: torch.device | None
    _model: jit.ScriptModule | None
    _transforms: Compose | None
    _output_classes: Sequence[str] | None
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
        self._device = None
        self._model = None
        self.output_classes = None
        self.is_initialized = False
        self._transforms = None
        self._executor = ThreadPoolExecutor(max_workers=max(max_workers, 1))

        # Setup logging
        self._logger = logging.getLogger(__name__)

    def configure(self):
        self._load_classes()

        self._setup_device()

        # Load and optimize model
        self._load_model()

        self._setup_transforms(layer_configs=self._config.preprocessing)

        self.is_initialized = True
        self._logger.info(f"Recognizer loaded successfully on {self._device}.")

    def _load_classes(self):
        path = os.path.join(get_config_folder_path(), self._config.output_config_path)
        output = RecognizerOutput.model_validate_json(Path(path).read_bytes())
        self._output_classes = [desc.name for desc in output.classes]

    def _setup_device(self):
        """Set up the computation device"""
        device = self._config.device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                self._logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                self._logger.info("Using CPU for inference.")

        self._device = torch.device(device)

    def _load_model(self):
        """Load the TorchScript model"""
        path = os.path.join(get_config_folder_path(), self._config.path)

        self._logger.info(f"Loading PyTorch model from path: {path}")

        try:
            # Load model
            model = jit.load(path, map_location=self._device)
            self._model = jit.optimize_for_inference(model)

            # Set to evaluation mode
            self._model.eval()

            # Move to a device
            self._model = self._model.to(self._device)
        except Exception as e:
            self._logger.error(f"Failed to load model: {self._config.path}")
            raise RuntimeError(e)

    def _setup_transforms(self, layer_configs: list[ImagePreprocessingConfiguration]):
        """Setup image preprocessing transforms"""
        layers = map(get_transform_layer, layer_configs)
        self._transforms = Compose(transforms=[ToTensor(), *layers])

    def preprocess_image(self, image: np.ndarray | Image) -> torch.Tensor:
        """
        Preprocess a single image

        Returns:
            Preprocessed tensor

        Raises:
            RuntimeError: If ``self._transforms`` is not configured.
        """
        if self._transforms is None:
            raise RuntimeError("Image preprocessing transforms have not configured.")

        # Apply transforms
        tensor = torch.unsqueeze(self._transforms(image), dim=0)
        return tensor.to(self._device)

    def predict(self,
                image: str | np.ndarray | Image,
                use_min_probability: bool = True, **kwargs) -> RecognizingResult:
        """
        Predict the classes and their respective probabilities for the given image
        by using the configured model. The method can handle images represented
        as a file path, a URL, or a preloaded image object.
        Returns a ``RecognizingResult`` object containing the
        recognized classes, their probabilities, and the inference time.

        :param image: Input image file path (str), URL (str), or an instance of
            ``np.ndarray`` or ``PIL.Image`` representing the image to be recognized.
        :param use_min_probability: If True, filters results, according to the minimum
            class probability threshold defined in the configuration. Defaults to True.
        :return: An instance of ``RecognizingResult`` containing recognized classes,
            probabilities, and inference time. The output results are filtered and
            limited by configuration settings.
        :raises RuntimeError: If the recognizer is not properly initialized before calling
            this method.
        :raises FileNotFoundError: If the image cannot be correctly processed into a compatible format.
        :raises HTTPError: If the provided URL fails to fetch the image.
        """
        if not self.is_initialized or self._model is None:
            raise RuntimeError("Recognizer is not properly initialized.")

        model = self._model
        start_time = time.time()

        # Preprocess
        if isinstance(image, str):
            if is_web_path(image):
                response = requests.get(image)
                response.raise_for_status()
                byte_data = BytesIO(response.content)
                image = Image.open(fp=byte_data, mode="r").convert(mode="RGB")
            else:
                image = Image.open(fp=image, mode="r").convert(mode="RGB")
        input_tensor = self.preprocess_image(image)

        # Inference
        with torch.no_grad():
            output: torch.Tensor = model(input_tensor)
        flatten_probs: list[float] = torch.sigmoid(output).cpu().numpy().flatten().tolist()

        zip_result = zip(self._output_classes, flatten_probs)
        if use_min_probability:
            min_prob = self._config.min_probability
            result: list[tuple[str, float]] = list(filter(lambda e: e[1] >= min_prob, zip_result))
        else:
            result = list(zip_result)

        probabilities: list[float] = []
        classes: list[str] = []
        for r in result:
            classes.append(r[0])
            probabilities.append(r[1])

        max_results = self._config.max_results
        return RecognizingResult(probabilities=probabilities[:max_results],
                                 classes=classes[:max_results],
                                 inference_time=time.time() - start_time)

    async def async_predict(self,
                            image: str | np.ndarray | Image,
                            use_min_probability: bool = True, **kwargs) -> RecognizingResult:
        """
        Asynchronously predict the classes and their respective probabilities for the given image
        by using the configured model. The method can handle images represented
        as a file path, a URL, or a preloaded image object.
        Returns a ``RecognizingResult`` object containing the
        recognized classes, their probabilities, and the inference time.

        :param image: Input image file path (str), URL (str), or an instance of
            ``np.ndarray`` or ``PIL.Image`` representing the image to be recognized.
        :param use_min_probability: If True, filters results, according to the minimum
            class probability threshold defined in the configuration. Defaults to True.
        :return: An instance of ``RecognizingResult`` containing recognized classes,
            probabilities, and inference time. The output results are filtered and
            limited by configuration settings.
        :raises RuntimeError: If the recognizer is not properly initialized before calling
            this method.
        :raises FileNotFoundError: If the image cannot be correctly processed into a compatible format.
        :raises HTTPError: If the provided URL fails to fetch the image.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.predict, image, use_min_probability)
