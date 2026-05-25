from .resnet32_cifar import ResNet32, resnet32
from .resnet32_blockwidth import BlockWidthResNet32, block_width_resnet32, DEFAULT_BLOCK_CHANNELS

__all__ = ["ResNet32", "resnet32", "BlockWidthResNet32", "block_width_resnet32", "DEFAULT_BLOCK_CHANNELS"]
