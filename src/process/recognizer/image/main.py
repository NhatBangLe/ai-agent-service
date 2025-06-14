import asyncio
import logging
import os.path
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from os import PathLike
from typing import Any

import jsonpickle
import numpy as np
import torch
import torch.jit as jit
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop, Pad, Grayscale

from src.config.model.recognizer.image import ImagePreprocessingConfiguration, ImageRecognizerConfiguration
from src.config.model.recognizer.image.preprocessing import ImageResizeConfiguration, \
    ImageNormalizeConfiguration, ImageCenterCropConfiguration, ImagePadConfiguration, ImageGrayscaleConfiguration
from src.process.recognizer.main import Recognizer, RecognizerOutput, RecognizingResult
from src.util.function import get_config_folder_path


def get_transform_layer(config: ImagePreprocessingConfiguration) -> torch.nn.Module:
    if isinstance(config, ImageResizeConfiguration):
        resize = typing.cast(ImageResizeConfiguration, config)

        return Resize(
            size=(resize.target_size,),
            interpolation=resize.interpolation,
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
    _model: jit.ScriptModule | None = None
    _transforms: Compose | None
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
        self._device = None
        self._model = None
        self.num_classes = None
        self.is_initialized = False
        self._transforms = None
        self._executor = ThreadPoolExecutor(max_workers=max(max_workers, 1))

        # Setup logging
        self._logger = logging.getLogger(__name__)

    def configure(self):
        self._setup_device()

        # Load and optimize model
        self._load_model()

        # Load output classes
        path = os.path.join(get_config_folder_path(), self._config.output_config_path)
        with open(path, "r") as config_file:
            json = config_file.read()
        output = RecognizerOutput.model_validate(jsonpickle.decode(json))
        self.num_classes = len(output.classes)

        self._setup_transforms(layer_configs=self._config.preprocessing)

        self.is_initialized = True
        self._logger.info(f"Recognizer loaded successfully on {self._device}.")
        self._logger.info(f"Number of classes: {self.num_classes}.")

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

    def _filter_multilabel_predictions(self, logits: torch.Tensor):
        """
        For multi-label classification with sigmoid activation
        """
        probs = torch.sigmoid(logits)  # Each class independent
        mask = probs >= self._config.min_probability

        active_classes = torch.nonzero(mask, as_tuple=True)[0]
        active_probs = probs[mask]

        return active_probs, active_classes

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
        probs, classes = self._filter_multilabel_predictions(output)

        return {
            'probabilities': probs.cpu().numpy().flatten().tolist(),
            'classes': classes.cpu().numpy().flatten().tolist(),
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
