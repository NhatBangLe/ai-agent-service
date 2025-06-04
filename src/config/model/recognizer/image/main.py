import asyncio
import logging
import os.path
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from typing import Any, Literal

import cv2
import jsonpickle
import numpy as np
import torch
import torch.jit as jit
import torch.nn.functional as functional
from PIL import Image
from pydantic import Field, BaseModel
from torchvision.transforms import Compose, Resize

from src.config.main import get_config_folder_path
from src.config.model.recognizer.image.preprocessing import ImageResizeConfiguration
from src.config.model.recognizer.main import RecognizerConfiguration, Recognizer, RecognizingResult, RecognizerOutput


class ImagePreprocessingConfiguration(BaseModel):
    """
    An interface for pre-processing image subclasses.
    """


# noinspection PyNestedDecorators
class ImageRecognizerConfiguration(RecognizerConfiguration):
    """
    An interface for image recognizer subclasses.
    """
    preprocessing: list[ImagePreprocessingConfiguration] | None = Field(default=None)
    output_config_path: str


class ImageRecognizer(Recognizer):
    """
    PyTorch inference class with optimizations
    """
    model_path: str
    _device: torch.device
    _model: jit.ScriptModule | None = None
    _input_size: tuple[int, int, int]
    _transforms: Compose
    num_classes: int | None
    is_initialized: bool
    _executor: ThreadPoolExecutor
    _logger: Logger

    def __init__(self,
                 model_path: str,
                 /,
                 device: Literal["auto", "cpu", "cuda"] = 'auto',
                 max_workers: int = 4,
                 **data: Any):
        """
        Initialize the inference engine

        Args:
            model_path: Path to the PyTorch model file
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            num_classes: Number of output classes (auto-detected if None)
            max_workers: Number of worker threads for async processing, min value 1
        """
        super().__init__(**data)
        self.model_path = model_path
        self._device = self._setup_device(device)
        self._model = None
        self.num_classes = None
        self.is_initialized = False
        self._executor = ThreadPoolExecutor(max_workers=max(max_workers, 1))

        # Setup logging
        self._logger = logging.getLogger(__name__)

    def configure(self, config: ImageRecognizerConfiguration):
        # Load and optimize model
        self._load_model()

        # Load output classes
        path = os.path.join(get_config_folder_path(), config.output_config_path)
        with open(path, "r") as config_file:
            json = config_file.read()
        output = RecognizerOutput.model_validate(jsonpickle.decode(json))
        self.num_classes = len(output.classes)

        self._setup_transforms(layer_configs=config.preprocessing)

        self.is_initialized = True
        self._logger.info(f"Recognizer loaded successfully on {self._device}.")
        self._logger.info(f"Input size: {self._input_size}.")
        self._logger.info(f"Number of classes: {self.num_classes}.")

    def _setup_device(self, device: Literal["auto", "cpu", "cuda"]) -> torch.device:
        """Set up the computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                self._logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                self._logger.info("Using CPU for inference.")

        return torch.device(device)

    def _load_model(self):
        """Load the TorchScript model"""
        self._logger.info(f"Loading PyTorch model from path: {self.model_path}")

        try:
            # Load model
            model = jit.load(self.model_path, map_location=self._device)
            self._model = jit.optimize_for_inference(model)

            # Set to evaluation mode
            self._model.eval()

            # Move to a device
            self._model = self._model.to(self._device)
        except Exception as e:
            self._logger.error(f"Failed to load model: {self.model_path}")
            raise RuntimeError(e)

    def _setup_transforms(self, layer_configs: list[ImagePreprocessingConfiguration]):
        """Setup image preprocessing transforms"""
        layers = map(self._get_transform_layer, layer_configs)
        self._transforms = Compose(transforms=layers)

    def _get_transform_layer(self, config: ImagePreprocessingConfiguration) -> torch.nn.Module:
        if isinstance(config, ImageResizeConfiguration):
            resize = typing.cast(ImageResizeConfiguration, config)
            width, height = resize.target_size

            return Resize(
                size=(height, width),
                interpolation=resize.interpolation,
            )
        else:
            raise NotImplementedError(f'Image preprocessing layer: {type(config)} is not supported.')

    def preprocess_image(self, image: str | np.ndarray | Image.Image) -> torch.Tensor:
        """
        Preprocess a single image

        Args:
            image: Image path, numpy array, or PIL Image

        Returns:
            Preprocessed tensor
        """
        # Load image if a path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB of necessary
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        # Apply transforms
        tensor = self._transforms(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(self._device)

    def predict(self,
                image: str | np.ndarray | Image.Image,
                return_probabilities: bool = True,
                top_k: int = 5) -> RecognizingResult:
        if not self.is_initialized or self._model is None:
            raise RuntimeError("Recognizer not properly initialized.")

        model = self._model
        start_time = time.time()

        # Preprocess
        input_tensor = self.preprocess_image(image)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)

        # Post-process
        if return_probabilities:
            probabilities = functional.softmax(output, dim=1)
            probs, indices = torch.topk(probabilities, top_k)

            result: RecognizingResult = {
                'predictions': indices.cpu().numpy().flatten().tolist(),
                'probabilities': probs.cpu().numpy().flatten().tolist(),
                'inference_time': time.time() - start_time
            }
        else:
            _, predicted = torch.max(output, 1)
            result: RecognizingResult = {
                'predictions': [predicted.cpu().item()],
                'probabilities': [1.0],
                'inference_time': time.time() - start_time
            }

        return result

    async def async_predict(self,
                            image: str | np.ndarray | Image.Image,
                            return_probabilities: bool = True,
                            top_k: int = 5) -> RecognizingResult:
        """
        Asynchronous prediction on a single image
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.predict,
            image,
            return_probabilities,
            top_k
        )
