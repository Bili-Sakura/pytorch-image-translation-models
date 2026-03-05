"""Common image transforms for translation tasks."""

from __future__ import annotations

from torchvision import transforms


def get_transforms(
    load_size: int = 286,
    crop_size: int = 256,
    flip: bool = True,
    normalize: bool = True,
    mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    std: tuple[float, ...] = (0.5, 0.5, 0.5),
) -> transforms.Compose:
    """Build a configurable transform pipeline.

    Parameters
    ----------
    load_size:
        Size to resize the shorter edge to before cropping.
    crop_size:
        Square crop size.
    flip:
        Whether to apply random horizontal flipping.
    normalize:
        Whether to normalize to [-1, 1].
    mean:
        Channel-wise mean for normalisation.
    std:
        Channel-wise std for normalisation.
    """
    pipeline: list[transforms.transforms.Transform] = [
        transforms.Resize(load_size, transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(crop_size),
    ]
    if flip:
        pipeline.append(transforms.RandomHorizontalFlip())
    pipeline.append(transforms.ToTensor())
    if normalize:
        pipeline.append(transforms.Normalize(mean, std))
    return transforms.Compose(pipeline)


def default_transforms(
    image_size: int = 256,
    normalize: bool = True,
) -> transforms.Compose:
    """Deterministic transform for evaluation / inference.

    Parameters
    ----------
    image_size:
        Target image size (square).
    normalize:
        Whether to normalize to [-1, 1].
    """
    pipeline: list[transforms.transforms.Transform] = [
        transforms.Resize(
            (image_size, image_size), transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
    ]
    if normalize:
        pipeline.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(pipeline)
