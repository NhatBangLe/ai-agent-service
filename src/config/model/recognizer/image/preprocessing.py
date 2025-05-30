from email.policy import default
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.config.model.recognizer.image.main import ImagePreprocessingConfiguration


class InterpolationType(str, Enum):
    """Supported interpolation methods for resizing"""
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    LANCZOS = "lanczos"
    AREA = "area"


class PaddingMode(str, Enum):
    """Padding modes for resizing with aspect ratio preservation"""
    CONSTANT = "constant"
    EDGE = "edge"
    REFLECT = "reflect"


class NormalizationType(str, Enum):
    """Supported normalization methods for classification"""
    MINMAX = "minmax"
    ZSCORE = "zscore"
    IMAGENET = "imagenet"
    CUSTOM = "custom"


class ColorSpace(str, Enum):
    """Supported color spaces"""
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"


# noinspection PyNestedDecorators
class ImageResizeConfiguration(ImagePreprocessingConfiguration):
    """
    Configuration for image resizing and associated operations.
    This step ensures images are scaled to a consistent size, which is
    often a requirement for input into deep learning models.
    """
    target_size: tuple[int, int] = Field(
        ...,  # This makes the field mandatory
        description="The target dimensions for the image after resizing, specified as a (width, height) tuple. "
                    "This size is typically determined by the input requirements of the classification model."
    )
    interpolation: InterpolationType = Field(
        default=InterpolationType.LINEAR,
        description="The interpolation algorithm to use for resizing the image. "
                    "Common choices include 'nearest' (fastest, can be pixelated), "
                    "'linear' (bi-linear, good balance of speed and quality), "
                    "and 'cubic' (bi-cubic, higher quality but slower). "
                    "Defaults to 'linear'."
    )
    maintain_aspect_ratio: bool = Field(
        default=True,
        description="If set to `True`, the original aspect ratio of the image will be preserved during resizing. "
                    "If `target_size` dimensions do not match the aspect ratio, padding or cropping "
                    "will be applied based on `padding_mode` or `center_crop_after_resize` to fit."
                    "If `False`, the image will be stretched or squashed to fit `target_size`."
    )
    padding_mode: PaddingMode = Field(
        default=PaddingMode.CONSTANT,
        description="Specifies how to fill the areas introduced by maintaining the aspect ratio "
                    "when `maintain_aspect_ratio` is `True`. "
                    "'constant' fills with `padding_color`. "
                    "'edge' extends the values from the image border. "
                    "'reflect' reflects pixels along the edge. "
                    "'symmetric' reflects pixels across the edge."
    )
    padding_color: tuple[int, int, int] = Field(
        default=(0, 0, 0),
        description="The RGB color (0-255 tuple, e.g., (0, 0, 0) for black) to use for padding "
                    "when `padding_mode` is 'constant' and `maintain_aspect_ratio` is `True`."
    )
    center_crop_after_resize: bool = Field(
        default=False,
        description="If set to `True`, after resizing (and potentially maintaining aspect ratio "
                    "with padding), the image will be centrally cropped to exactly match the "
                    "`target_size`. This is an alternative to padding for filling the target dimensions "
                    "when aspect ratio is maintained. This field takes precedence over `padding_mode` "
                    "if both `maintain_aspect_ratio` and `center_crop_after_resize` are `True`."
    )

    @field_validator('target_size', mode="after")
    @classmethod
    def validate_target_size(cls, v: tuple[int, int]):
        if len(v) != 2 or any(dim <= 0 for dim in v):
            raise ValueError("target_size must be a tuple of 2 positive integers.")
        return v

    @field_validator('padding_color', mode="after")
    @classmethod
    def validate_padding_color(cls, color: tuple[int, int, int]):
        if len(color) != 3 or any(c < 0 for c in color):
            raise ValueError("padding_color must be a tuple of 3 non negative integers.")
        return color


# noinspection PyNestedDecorators
class ImageNormalizeConfiguration(ImagePreprocessingConfiguration):
    method: NormalizationType = NormalizationType.IMAGENET
    custom_mean: list[float] | None = None
    custom_std: list[float] | None = None
    pixel_range: tuple[float, float] = (0.0, 1.0)
    scale_first: bool = True

    @field_validator('custom_mean', 'custom_std')
    @classmethod
    def validate_custom_params(cls, v, values):
        if values.get('method') == NormalizationType.CUSTOM and v is None:
            raise ValueError("custom_mean and custom_std required when method is 'custom'.")
        if v is not None and len(v) not in [1, 3]:
            raise ValueError("custom_mean and custom_std must have 1 or 3 values.")
        return v


# noinspection PyNestedDecorators
class CropConfig(ImagePreprocessingConfiguration):
    crop_type: str = Field(default="center", pattern="^(center|custom)$")
    crop_size: tuple[int, int] | None = None
    custom_coords: tuple[int, int, int, int] = None  # (x, y, width, height)

    @field_validator('custom_coords')
    @classmethod
    def validate_custom_coords(cls, v, values):
        if values.get('crop_type') == 'custom' and v is None:
            raise ValueError("custom_coords required when crop_type is 'custom'")
        return v


class ColorAdjustmentConfig(BaseModel):
    """Configuration for color adjustments for classification"""
    enabled: bool = False
    target_color_space: ColorSpace = ColorSpace.RGB
    brightness_adjustment: float = Field(0.0, ge=-0.5, le=0.5)
    contrast_adjustment: float = Field(1.0, gt=0.5, le=2.0)
    gamma_correction: float | None = Field(None, gt=0.5, le=2.0)


class FilterConfig(BaseModel):
    """Configuration for image filtering and enhancement"""
    enabled: bool = False
    gaussian_blur: float | None = Field(None, gt=0.0, le=5.0)
    sharpen: bool = False
    noise_reduction: bool = False


# noinspection PyNestedDecorators
class CustomPreprocessingStep(BaseModel):
    """Configuration for custom preprocessing steps"""
    name: str = Field(..., min_length=1)
    function_name: str = Field(..., min_length=1)
    parameters: dict[str, Any] = {}
    enabled: bool = True
    order: int = Field(999, ge=0, le=9999)
    description: str | None = None

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Name must contain only alphanumeric characters, underscores, and hyphens")
        return v


class TensorConfig(BaseModel):
    """Configuration for tensor output formatting"""
    output_format: str = Field("numpy", pattern="^(numpy|torch|tensorflow)$")
    data_type: str = Field("float32", pattern="^(float16|float32|float64)$")
    channel_order: str = Field("CHW", pattern="^(CHW|HWC)$")
    add_batch_dimension: bool = True
    device: str | None = None  # For torch: 'cpu', 'cuda'


# noinspection PyNestedDecorators
class CNNClassificationPreprocessingConfig(BaseModel):
    """
    Configuration for CNN image classification preprocessing pipeline.
    
    This configuration class provides a flexible and extensible way to define
    image preprocessing steps specifically for CNN classification models
    with full validation support.
    """

    # Model specifications
    model_input_size: tuple[int, int] = Field(..., description="Expected model input size (width, height)")
    input_channels: int = Field(3, ge=1, le=4, description="Expected input channels (1=grayscale, 3=RGB)")
    model_name: str | None = Field(None, description="Classification model identifier")
    num_classes: str | None = Field(None, gt=0, description="Number of classification classes")

    # Core preprocessing configurations
    resize: ImageResizeConfiguration
    normalization: ImageNormalizeConfiguration = ImageNormalizeConfiguration()
    color_adjustment: ColorAdjustmentConfig = ColorAdjustmentConfig()
    crop: CropConfig = CropConfig()
    filter: FilterConfig = FilterConfig()

    # Custom preprocessing steps
    custom_steps: list[CustomPreprocessingStep] = []

    # Pipeline settings
    processing_order: list[str] = Field(
        default=["crop", "color_adjustment", "filter", "resize", "normalization"],
        description="Order of preprocessing steps"
    )

    # Performance settings
    batch_processing: bool = True
    max_batch_size: int = Field(32, ge=1, le=512)
    num_workers: int = Field(1, ge=1, le=8)

    # Output configuration
    tensor_config: TensorConfig = TensorConfig()

    # Quality settings
    quality_checks: bool = True
    log_processing_time: bool = False

    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "forbid"
        schema_extra = {
            "example": {
                "model_input_size": [224, 224],
                "input_channels": 3,
                "model_name": "ResNet50",
                "num_classes": 1000,
                "resize": {
                    "enabled": True,
                    "target_size": [224, 224],
                    "interpolation": "linear",
                    "maintain_aspect_ratio": True
                },
                "normalization": {
                    "enabled": True,
                    "method": "imagenet"
                },
                "tensor_config": {
                    "output_format": "torch",
                    "data_type": "float32",
                    "channel_order": "CHW"
                }
            }
        }

    def __init__(self, **data):
        # Autoconfigure resize target_size based on model_input_size if not provided
        if 'resize' not in data and 'model_input_size' in data:
            data['resize'] = ImageResizeConfiguration(
                enabled=True,
                target_size=data['model_input_size']
            )
        elif 'resize' in data and isinstance(data['resize'], dict) and 'target_size' not in data['resize']:
            data['resize']['target_size'] = data.get('model_input_size', (224, 224))

        super().__init__(**data)

    @field_validator('processing_order')
    @classmethod
    def validate_processing_order(cls, v):
        """Ensure all specified steps are valid"""
        valid_steps = {
            "crop", "color_adjustment", "filter", "resize", "normalization", "custom"
        }
        invalid_steps = set(v) - valid_steps
        if invalid_steps:
            raise ValueError(f"Invalid processing steps: {invalid_steps}")
        return v

    @field_validator('custom_steps')
    @classmethod
    def validate_custom_steps(cls, v):
        """Ensure custom steps have unique names"""
        names = [step.name for step in v]
        if len(names) != len(set(names)):
            raise ValueError("Custom step names must be unique")
        return v

    @field_validator('model_input_size')
    @classmethod
    def validate_model_input_size(cls, v):
        if len(v) != 2 or any(dim <= 0 for dim in v):
            raise ValueError("model_input_size must be a tuple of 2 positive integers")
        return v

    def get_enabled_steps(self) -> list[str]:
        """Get list of enabled preprocessing steps in processing order"""
        enabled_steps = []

        for step_name in self.processing_order:
            if step_name == "custom":
                # Add enabled custom steps sorted by order
                enabled_custom = [step for step in self.custom_steps if step.enabled]
                enabled_custom.sort(key=lambda x: x.order)
                enabled_steps.extend([f"custom_{step.name}" for step in enabled_custom])
            else:
                step_config = getattr(self, step_name, None)
                if step_config and getattr(step_config, 'enabled', False):
                    enabled_steps.append(step_name)

        return enabled_steps

    def add_custom_step(self, name: str, function_name: str,
                        parameters: dict[str, Any] = None,
                        order: int = 999, enabled: bool = True,
                        description: str = None) -> None:
        """Add a custom preprocessing step for classification"""
        custom_step = CustomPreprocessingStep(
            name=name,
            function_name=function_name,
            parameters=parameters or {},
            order=order,
            enabled=enabled,
            description=description
        )
        self.custom_steps.append(custom_step)

    def enable_step(self, step_name: str) -> None:
        """Enable a preprocessing step"""
        if hasattr(self, step_name):
            step_config = getattr(self, step_name)
            if hasattr(step_config, 'enabled'):
                step_config.enabled = True
        else:
            for step in self.custom_steps:
                if step.name == step_name:
                    step.enabled = True
                    break

    def disable_step(self, step_name: str) -> None:
        """Disable a preprocessing step"""
        if hasattr(self, step_name):
            step_config = getattr(self, step_name)
            if hasattr(step_config, 'enabled'):
                step_config.enabled = False
        else:
            for step in self.custom_steps:
                if step.name == step_name:
                    step.enabled = False
                    break

    def get_classification_summary(self) -> dict[str, Any]:
        """Get a summary of the classification preprocessing configuration"""
        return {
            "model_name": self.model_name,
            "model_input_size": self.model_input_size,
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "enabled_steps": self.get_enabled_steps(),
            "resize_config": {
                "target_size": self.resize.target_size,
                "maintain_aspect_ratio": self.resize.maintain_aspect_ratio,
                "interpolation": self.resize.interpolation
            } if self.resize.enabled else None,
            "normalization_method": self.normalization.method if self.normalization.enabled else None,
            "output_format": self.tensor_config.output_format,
            "output_dtype": self.tensor_config.data_type,
            "channel_order": self.tensor_config.channel_order,
            "batch_processing": self.batch_processing,
            "max_batch_size": self.max_batch_size,
            "custom_steps_count": len([s for s in self.custom_steps if s.enabled])
        }

    def get_expected_output_shape(self, batch_size: int = 1) -> tuple[int, ...]:
        """Get the expected output tensor shape after preprocessing"""
        h, w = self.model_input_size
        c = self.input_channels

        if self.tensor_config.channel_order == "CHW":
            shape = (c, h, w)
        else:  # HWC
            shape = (h, w, c)

        if self.tensor_config.add_batch_dimension:
            shape = (batch_size,) + shape

        return shape

    def validate_classification_pipeline(self) -> list[str]:
        """Validate the classification preprocessing pipeline and return warnings"""
        warnings = []

        # Check if resize target matches model input size
        if self.resize.enabled and self.resize.target_size != self.model_input_size:
            warnings.append(
                f"Resize target {self.resize.target_size} doesn't match model input size {self.model_input_size}")

        # Check normalization compatibility for classification
        if self.normalization.enabled and self.normalization.method == NormalizationType.IMAGENET:
            if self.input_channels != 3:
                warnings.append("ImageNet normalization is designed for 3-channel RGB images")

        # Check color space and channels compatibility
        if self.color_adjustment.enabled:
            if self.color_adjustment.target_color_space == ColorSpace.GRAY and self.input_channels != 1:
                warnings.append("Grayscale color space selected but input_channels != 1")
            elif self.color_adjustment.target_color_space in [ColorSpace.RGB,
                                                              ColorSpace.BGR] and self.input_channels != 3:
                warnings.append(
                    f"{self.color_adjustment.target_color_space} color space selected but input_channels != 3")

        # Check crop size compatibility
        if self.crop.enabled and self.crop.crop_size:
            crop_w, crop_h = self.crop.crop_size
            target_w, target_h = self.model_input_size
            if crop_w < target_w or crop_h < target_h:
                warnings.append(
                    f"Crop size {self.crop.crop_size} is smaller than model input size {self.model_input_size}")

        return warnings

    def get_preprocessing_stats(self) -> dict[str, int]:
        """Get statistics about the preprocessing configuration"""
        return {
            "total_steps": len(self.processing_order),
            "enabled_steps": len(self.get_enabled_steps()),
            "custom_steps": len(self.custom_steps),
            "enabled_custom_steps": len([s for s in self.custom_steps if s.enabled]),
            "input_size_pixels": self.model_input_size[0] * self.model_input_size[1],
            "input_channels": self.input_channels
        }


# Factory functions for common classification scenarios
class ClassificationPreprocessingFactory:
    """Factory class for creating common classification preprocessing configurations"""

    @staticmethod
    def create_imagenet_config(
            model_input_size: tuple[int, int] = (224, 224),
            model_name: str = None
    ) -> CNNClassificationPreprocessingConfig:
        """Create ImageNet-style preprocessing configuration"""
        return CNNClassificationPreprocessingConfig(
            model_input_size=model_input_size,
            model_name=model_name,
            num_classes=1000,
            input_channels=3,
            resize=ImageResizeConfiguration(
                enabled=True,
                target_size=model_input_size,
                interpolation=InterpolationType.LINEAR,
                maintain_aspect_ratio=True,
                center_crop_after_resize=True
            ),
            normalization=ImageNormalizeConfiguration(
                enabled=True,
                method=NormalizationType.IMAGENET
            ),
            tensor_config=TensorConfig(
                output_format="torch",
                data_type="float32",
                channel_order="CHW"
            )
        )

    @staticmethod
    def create_cifar_config(
            model_name: str = None
    ) -> CNNClassificationPreprocessingConfig:
        """Create CIFAR-style preprocessing configuration"""
        return CNNClassificationPreprocessingConfig(
            model_input_size=(32, 32),
            model_name=model_name,
            num_classes=10,
            input_channels=3,
            resize=ImageResizeConfiguration(
                enabled=True,
                target_size=(32, 32),
                interpolation=InterpolationType.LINEAR,
                maintain_aspect_ratio=False
            ),
            normalization=ImageNormalizeConfiguration(
                enabled=True,
                method=NormalizationType.CUSTOM,
                custom_mean=[0.4914, 0.4822, 0.4465],
                custom_std=[0.2023, 0.1994, 0.2010]
            ),
            tensor_config=TensorConfig(
                output_format="torch",
                data_type="float32",
                channel_order="CHW"
            )
        )

    @staticmethod
    def create_grayscale_config(
            model_input_size: tuple[int, int] = (224, 224),
            model_name: str = None,
            num_classes: int = None
    ) -> CNNClassificationPreprocessingConfig:
        """Create grayscale classification preprocessing configuration"""
        return CNNClassificationPreprocessingConfig(
            model_input_size=model_input_size,
            model_name=model_name,
            num_classes=num_classes,
            input_channels=1,
            color_adjustment=ColorAdjustmentConfig(
                enabled=True,
                target_color_space=ColorSpace.GRAY
            ),
            resize=ImageResizeConfiguration(
                enabled=True,
                target_size=model_input_size,
                interpolation=InterpolationType.LINEAR,
                maintain_aspect_ratio=True
            ),
            normalization=ImageNormalizeConfiguration(
                enabled=True,
                method=NormalizationType.MINMAX
            ),
            tensor_config=TensorConfig(
                output_format="numpy",
                data_type="float32",
                channel_order="CHW"
            )
        )

    @staticmethod
    def create_mobile_classification_config(
            model_input_size: tuple[int, int] = (224, 224),
            model_name: str = None,
            num_classes: int = 1000
    ) -> CNNClassificationPreprocessingConfig:
        """Create mobile-optimized classification preprocessing configuration"""
        return CNNClassificationPreprocessingConfig(
            model_input_size=model_input_size,
            model_name=model_name,
            num_classes=num_classes,
            input_channels=3,
            resize=ImageResizeConfiguration(
                enabled=True,
                target_size=model_input_size,
                interpolation=InterpolationType.LINEAR,
                maintain_aspect_ratio=True
            ),
            normalization=ImageNormalizeConfiguration(
                enabled=True,
                method=NormalizationType.MINMAX,
                pixel_range=(0.0, 1.0)
            ),
            tensor_config=TensorConfig(
                output_format="numpy",
                data_type="float32",
                channel_order="HWC"  # Many mobile frameworks prefer HWC
            ),
            batch_processing=False,
            max_batch_size=1,
            num_workers=1
        )

    @staticmethod
    def create_custom_config(
            model_input_size: tuple[int, int],
            model_name: str,
            num_classes: int,
            normalization_mean: list[float] = None,
            normalization_std: list[float] = None
    ) -> CNNClassificationPreprocessingConfig:
        """Create custom classification preprocessing configuration"""
        norm_config = ImageNormalizeConfiguration(enabled=True)

        if normalization_mean and normalization_std:
            norm_config.method = NormalizationType.CUSTOM
            norm_config.custom_mean = normalization_mean
            norm_config.custom_std = normalization_std
        else:
            norm_config.method = NormalizationType.IMAGENET

        return CNNClassificationPreprocessingConfig(
            model_input_size=model_input_size,
            model_name=model_name,
            num_classes=num_classes,
            input_channels=3,
            resize=ImageResizeConfiguration(
                enabled=True,
                target_size=model_input_size,
                interpolation=InterpolationType.LINEAR,
                maintain_aspect_ratio=True
            ),
            normalization=norm_config,
            tensor_config=TensorConfig(
                output_format="torch",
                data_type="float32",
                channel_order="CHW"
            )
        )
