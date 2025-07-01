import asyncio
import logging
import os
import time

import typing
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from os import PathLike
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import jit as jit
from torchvision.transforms import Resize, Normalize, CenterCrop, Pad, Grayscale, Compose, ToTensor

from src.config.model.recognizer.image import ImagePreprocessingConfiguration, ImageRecognizerConfiguration
from src.config.model.recognizer.image.preprocessing import ImageResizeConfiguration, ImageNormalizeConfiguration, \
    ImageCenterCropConfiguration, ImagePadConfiguration, ImageGrayscaleConfiguration, INTERPOLATION_MODE_DICT
from src.process.recognizer import Recognizer, RecognizingResult, RecognizerOutput
from src.util.function import get_config_folder_path

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

    def preprocess_image(self, image_path: str | bytes | PathLike[str] | PathLike[bytes]) -> torch.Tensor:
        """
        Preprocess a single image

        Returns:
            Preprocessed tensor

        Raises:
            RuntimeError: If ``self._transforms`` is not configured.
            FileNotFoundError: If the file cannot be found.
            PIL.UnidentifiedImageError: If the image cannot be opened and identified.
        """
        if self._transforms is None:
            raise RuntimeError("Image preprocessing transforms have not configured.")

        # Load image
        image = Image.open(fp=image_path, mode="r").convert(mode="RGB")

        # Apply transforms
        tensor = torch.unsqueeze(self._transforms(image), dim=0)
        return tensor.to(self._device)

    def predict(self,
                image: str | np.ndarray | Image.Image,
                use_min_probability: bool = True) -> RecognizingResult:
        if not self.is_initialized or self._model is None:
            raise RuntimeError("Recognizer not properly initialized.")

        model = self._model
        start_time = time.time()

        # Preprocess
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
        return {
            'probabilities': probabilities[:max_results],
            'classes': classes[:max_results],
            'inference_time': time.time() - start_time
        }

    async def async_predict(self,
                            image: str | np.ndarray | Image.Image,
                            use_min_probability: bool = True) -> RecognizingResult:
        """
        Asynchronous prediction on a single image
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.predict,
            image,
            use_min_probability
        )
