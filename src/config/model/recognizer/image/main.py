import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import torchvision.transforms as transforms
from PIL import Image
from pydantic import Field, BaseModel

from src.config.model.recognizer.main import RecognizerConfiguration, Recognizer, RecognizingResult


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
    batch_size: int
    device: torch.device
    model = None
    input_size: tuple[int, int, int]
    transforms: transforms.Compose
    num_classes: int | None
    is_initialized: bool
    max_workers: int
    executor: ThreadPoolExecutor
    logger: Logger

    def __init__(self,
                 model_path: str,
                 /,
                 device: str = 'auto',
                 batch_size: int = 1,
                 num_classes: int | None = None,
                 max_workers: int = 4,
                 **data: Any):
        """
        Initialize the inference engine

        Args:
            model_path: Path to the PyTorch model file
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            batch_size: Batch size for inference
            num_classes: Number of output classes (auto-detected if None)
            max_workers: Number of worker threads for async processing
        """
        super().__init__(**data)
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = self._setup_device(device)
        self.model = None
        self.num_classes = num_classes
        self.is_initialized = False
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Load and optimize model
        self._load_model()
        self._detect_input_size()
        self._setup_transforms()
        self._optimize_model()

        self.logger.info(f"Recognizer loaded successfully on {self.device}.")
        self.logger.info(f"Input size: {self.input_size}.")
        self.logger.info(f"Number of classes: {self.num_classes}.")

    def _setup_device(self, device: str) -> torch.device:
        """Set up the computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                self.logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                self.logger.info("Using CPU for inference.")

        return torch.device(device)

    def _load_model(self):
        """Load the PyTorch model"""
        try:
            # Load model
            self.model = torch.load(self.model_path, weights_only=False, map_location=self.device)

            # Handle different model formats
            if isinstance(self.model, dict):
                if 'model' in self.model:
                    self.model = self.model['model']
                elif 'state_dict' in self.model:
                    # Need architecture - this is more complex
                    raise ValueError("State dict format requires model architecture.")

            # Set to evaluation mode
            self.model.eval()

            # Move to a device
            self.model = self.model.to(self.device)

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _detect_input_size(self):
        """Automatically detect the expected input size"""
        common_sizes = [
            (3, 224, 224),  # ImageNet standard
            (3, 256, 256),  # Larger images
            (3, 299, 299),  # Inception
            (3, 32, 32),  # CIFAR-10
            (1, 28, 28),  # MNIST
            (3, 128, 128),  # Medium size
        ]

        input_size: tuple[int, int, int] | None = None
        for size in common_sizes:
            try:
                dummy_input = torch.randn(1, *size).to(self.device)
                with torch.no_grad():
                    output = self.model(dummy_input)
                input_size = size

                # Detect number of classes from output
                if self.num_classes is None:
                    if len(output.shape) == 2:  # (batch, classes)
                        self.num_classes = output.shape[1]
                    elif len(output.shape) == 4:  # (batch, classes, H, W) - segmentation
                        self.num_classes = output.shape[1]

                break
            except Exception as e:
                self.logger.warning(e)

        if input_size is None:
            raise ValueError("Could not determine input size automatically.")
        self.input_size = input_size

    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        # Standard ImageNet preprocessing
        if self.input_size[0] == 3:  # RGB
            self.transforms = transforms.Compose([
                transforms.Resize((self.input_size[1], self.input_size[2])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:  # Grayscale
            self.transforms = transforms.Compose([
                transforms.Resize((self.input_size[1], self.input_size[2])),
                transforms.Grayscale(num_output_channels=self.input_size[0]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def _optimize_model(self):
        """Apply optimizations"""
        # Disable gradient computation globally
        torch.set_grad_enabled(False)

        # JIT compilation for faster inference
        try:
            dummy_input = torch.randn(1, *self.input_size).to(self.device)
            self.model = torch.jit.trace(self.model, dummy_input)
            self.logger.info("Model JIT compiled successfully.")
        except Exception as e:
            self.logger.warning(f"JIT compilation failed: {e}.")

        # Set to evaluation mode again after JIT
        self.model.eval()

        self.is_initialized = True

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
            # if len(image.shape) == 3 and image.shape[2] == 3:
            #     # Convert BGR to RGB if needed
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        # Apply transforms
        tensor = self.transforms(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)

    def predict(self,
                image: str | np.ndarray | Image.Image,
                return_probabilities: bool = True,
                top_k: int = 5) -> RecognizingResult:
        if not self.is_initialized:
            raise RuntimeError("Recognizer not properly initialized.")

        start_time = time.time()

        # Preprocess
        input_tensor = self.preprocess_image(image)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)

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
            self.executor,
            self.predict,
            image,
            return_probabilities,
            top_k
        )

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'batch_size': self.batch_size,
            'max_workers': self.max_workers
        }
