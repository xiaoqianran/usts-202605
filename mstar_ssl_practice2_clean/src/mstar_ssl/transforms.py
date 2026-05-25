from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps


class Compose:
    def __init__(self, transforms: Sequence[object]):
        self.transforms = list(transforms)

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image


class Resize:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize((self.size, self.size), resample=Image.BILINEAR)


class RandomCrop:
    def __init__(self, size: int, padding: int = 0, fill: int = 0):
        self.size = size
        self.padding = padding
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.padding:
            image = ImageOps.expand(image, border=self.padding, fill=self.fill)
        width, height = image.size
        if width < self.size or height < self.size:
            canvas = Image.new("L", (max(width, self.size), max(height, self.size)), self.fill)
            canvas.paste(image, ((canvas.width - width) // 2, (canvas.height - height) // 2))
            image = canvas
            width, height = image.size
        left = random.randint(0, width - self.size)
        top = random.randint(0, height - self.size)
        return image.crop((left, top, left + self.size, top + self.size))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.mirror(image) if random.random() < self.p else image


class RandomAffine:
    def __init__(self, degrees: float = 10.0, translate: float = 0.08, p: float = 0.8):
        self.degrees = degrees
        self.translate = translate
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        max_dx = int(image.width * self.translate)
        max_dy = int(image.height * self.translate)
        return image.rotate(
            random.uniform(-self.degrees, self.degrees),
            resample=Image.BILINEAR,
            translate=(random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy)),
            fillcolor=0,
        )


class ToTensorNormalize:
    def __init__(self, mean: float = 0.5, std: float = 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[None, :, :]
        else:
            arr = arr.transpose(2, 0, 1)
        return (torch.from_numpy(arr) - self.mean) / self.std


class GaussianNoise:
    def __init__(self, std: float = 0.03):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, -1.0, 1.0)


class RandomErasing:
    def __init__(self, p: float = 0.25, scale: tuple[float, float] = (0.02, 0.08), value: float = 0.0):
        self.p = p
        self.scale = scale
        self.value = value

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        _, height, width = tensor.shape
        erase_area = random.uniform(*self.scale) * height * width
        side = max(1, int(erase_area ** 0.5))
        erase_h, erase_w = min(height, side), min(width, side)
        top = random.randint(0, height - erase_h)
        left = random.randint(0, width - erase_w)
        out = tensor.clone()
        out[:, top : top + erase_h, left : left + erase_w] = self.value
        return out


def build_transforms(img_size: int):
    eval_transform = Compose([Resize(img_size), ToTensorNormalize()])
    weak_transform = Compose(
        [Resize(img_size), RandomCrop(img_size, padding=8), RandomHorizontalFlip(), ToTensorNormalize()]
    )
    strong_transform = Compose(
        [
            Resize(img_size),
            RandomCrop(img_size, padding=12),
            RandomHorizontalFlip(),
            RandomAffine(degrees=10, translate=0.08),
            ToTensorNormalize(),
            GaussianNoise(std=0.03),
            RandomErasing(p=0.25),
        ]
    )
    return weak_transform, strong_transform, eval_transform

