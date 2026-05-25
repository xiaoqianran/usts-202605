from .resnet32_cifar import ResNet32, resnet32
from .resnet32_width import WidthResNet32, width_resnet32

__all__ = ["ResNet32", "resnet32", "WidthResNet32", "width_resnet32"]

from .resnet32_block_width import BlockWidthResNet32, block_width_resnet32, BASELINE_BLOCK_CHANNELS
__all__ = list(set(globals().get("__all__", [])) | {"BlockWidthResNet32", "block_width_resnet32", "BASELINE_BLOCK_CHANNELS"})
