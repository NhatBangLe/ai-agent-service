from enum import Enum
from typing import Sequence

from pydantic import Field, field_validator
from torchvision.transforms import InterpolationMode as TransformsInterpolationMode

from src.config.model.recognizer.image import ImagePreprocessingConfiguration


class InterpolationMode(str, Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


INTERPOLATION_MODE_DICT = {
    InterpolationMode.NEAREST: TransformsInterpolationMode.NEAREST,
    InterpolationMode.NEAREST_EXACT: TransformsInterpolationMode.NEAREST_EXACT,
    InterpolationMode.BILINEAR: TransformsInterpolationMode.BILINEAR,
    InterpolationMode.BICUBIC: TransformsInterpolationMode.BICUBIC,
}


class ImageResizeConfiguration(ImagePreprocessingConfiguration):
    """
    Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means a maximum of two leading dimensions
    """
    target_size: int = Field(
        ...,  # This makes the field mandatory
        description="The target dimensions for the image after resizing, specified as a (width, height) tuple. "
                    "This size is typically determined by the input requirements of the classification model.")
    interpolation: InterpolationMode = Field(
        default=InterpolationMode.BILINEAR,
        description="The interpolation algorithm to use for resizing the image. "
                    "Common choices include 'nearest' (fastest, can be pixelated), "
                    "'bi-linear' (good balance of speed and quality), "
                    "and 'bi-cubic' (higher quality but slower). "
                    "Defaults to 'bilinear'.")
    max_size: int | None = Field(
        default=None,
        description="The maximum allowed for the longer edge of "
                    "the resized image. If the longer edge of the image is greater "
                    "than `max_size` after being resized according to size, "
                    "`size` will be overruled so that the longer edge is equal to `max_size`. "
                    "As a result, the smaller edge may be shorter than `size`.")


class ImageNormalizeConfiguration(ImagePreprocessingConfiguration):
    """
    Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],.,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    Note: This transform acts out of place, i.e., it does not mutate the input tensor.
    """

    mean: Sequence[float] = Field(description="Sequence of means for each channel.")
    std: Sequence[float] = Field(description="Sequence of standard deviations for each channel.")
    inplace: bool | None = Field(default=False, description="Bool to make this operation in-place.")


class ImageCenterCropConfiguration(ImagePreprocessingConfiguration):
    """
    Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If the image size is smaller than the output size along any edge, the image is padded with 0 and then center cropped.
    """

    size: Sequence[int] | int = Field(
        description="Desired output size of the crop. If size is an int instead of "
                    "sequence like (h, w), a square crop (size, size) is made. "
                    "If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).")


class PaddingMode(str, Enum):
    """
    Type of padding. Should be: constant, edge, reflect, or symmetric.
    Default is constant.

    - CONSTANT: pads with a constant value, this value is specified with fill

    - EDGE: pads with the last value at the edge of the image.
      If input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

    - REFLECT: pads with reflection of the image without repeating the last value on the edge.
      For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflection mode
      will result in [3, 2, 1, 2, 3, 4, 3, 2]

    - SYMMETRIC: pads with reflection of image repeating the last value on the edge.
      For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
      will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    CONSTANT = "constant"
    EDGE = "edge"
    REFLECT = "reflect"
    SYMMETRIC = "symmetric"


class ImagePadConfiguration(ImagePreprocessingConfiguration):
    """
    Pad the given image on all sides with the given "pad" value.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means at most 2 leading dimensions for mode reflect and symmetric,
    at most 3 leading dimensions for mode edge,
    and an arbitrary number of leading dimensions for mode constant
    """

    padding: Sequence[int] | int = Field(
        description="Padding on each border. If a single int is provided this"
                    "is used to pad all borders. If sequence of length 2 is provided this is the padding"
                    "on left/right and top/bottom respectively. If a sequence of length 4 is provided"
                    "this is the padding for the left, top, right and bottom borders respectively.")
    fill: int | tuple[int, ...] = Field(
        default=0,
        description="Pixel fill value for constant fill. Default is 0. If a tuple of "
                    "length 3, it is used to fill R, G, B channels respectively. "
                    "This value is only used when the padding_mode is constant.")
    mode: PaddingMode = Field(
        default=PaddingMode.CONSTANT,
        description="Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.")


# noinspection PyNestedDecorators
class ImageGrayscaleConfiguration(ImagePreprocessingConfiguration):
    """
    Convert image to grayscale.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions
    """

    num_output_channels: int = Field(
        default=3, ge=1, le=3,
        description="(1 or 3) number of channels desired for output image. "
                    "If ``num_output_channels == 1`` : returned image is single channel. "
                    "If ``num_output_channels == 3`` : returned image is 3 channel with r == g == b.")

    @field_validator('num_output_channels', mode="after")
    @classmethod
    def validate_num_output_channels(cls, v: int):
        if v != 1 and v != 3:
            raise ValueError("num_output_channels must be 1 or 3.")
        return v
