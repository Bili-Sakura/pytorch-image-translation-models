"""Image datasets for paired and unpaired translation tasks."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class PairedImageDataset(Dataset):
    """Dataset for paired image-to-image translation.

    Expects a directory layout where source and target images share the
    same file names under separate sub-directories::

        root/
            source/
                img_001.png
                img_002.png
            target/
                img_001.png
                img_002.png

    Parameters
    ----------
    root:
        Root directory containing ``source`` and ``target`` folders.
    source_dir:
        Name of the source image sub-directory.
    target_dir:
        Name of the target image sub-directory.
    transform:
        Optional transform applied to **both** images consistently.
    source_transform:
        Optional additional transform applied to source images only.
    target_transform:
        Optional additional transform applied to target images only.
    extensions:
        Accepted file extensions.
    """

    EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(
        self,
        root: str | Path | None = None,
        source_dir: str = "source",
        target_dir: str = "target",
        root_source: str | Path | None = None,
        root_target: str | Path | None = None,
        transform: Callable | None = None,
        source_transform: Callable | None = None,
        target_transform: Callable | None = None,
        transform_source: Callable | None = None,
        transform_target: Callable | None = None,
        extensions: set[str] | None = None,
    ) -> None:
        if root_source is not None and root_target is not None:
            self.source_root = Path(root_source)
            self.target_root = Path(root_target)
            self.root = self.source_root.parent
        else:
            if root is None:
                raise ValueError("Provide either root or (root_source, root_target)")
            self.root = Path(root)
            self.source_root = self.root / source_dir
            self.target_root = self.root / target_dir

        src_tf = source_transform or transform_source
        tgt_tf = target_transform or transform_target
        self.transform = transform
        self.source_transform = src_tf
        self.target_transform = tgt_tf

        exts = extensions or self.EXTENSIONS

        self.filenames = sorted(
            f
            for f in os.listdir(self.source_root)
            if Path(f).suffix.lower() in exts
            and (self.target_root / f).exists()
        )

        if len(self.filenames) == 0:
            raise FileNotFoundError(
                f"No matching image pairs found under {self.source_root} "
                f"and {self.target_root}"
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        fname = self.filenames[index]
        source = Image.open(self.source_root / fname).convert("RGB")
        target = Image.open(self.target_root / fname).convert("RGB")

        if self.transform is not None:
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            torch.manual_seed(seed)
            source = self.transform(source)
            random.seed(seed)
            torch.manual_seed(seed)
            target = self.transform(target)

        if self.source_transform is not None:
            source = self.source_transform(source)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"source": source, "target": target, "filename": fname}


class UnpairedImageDataset(Dataset):
    """Dataset for unpaired image-to-image translation (e.g. CycleGAN).

    Loads images from two independent directories without requiring
    matching file names.

    Parameters
    ----------
    root_a:
        Directory containing domain-A images.
    root_b:
        Directory containing domain-B images.
    transform_a:
        Transform applied to domain-A images.
    transform_b:
        Transform applied to domain-B images.
    extensions:
        Accepted file extensions.
    """

    EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(
        self,
        root_a: str | Path,
        root_b: str | Path,
        transform_a: Callable | None = None,
        transform_b: Callable | None = None,
        extensions: set[str] | None = None,
    ) -> None:
        self.root_a = Path(root_a)
        self.root_b = Path(root_b)
        self.transform_a = transform_a
        self.transform_b = transform_b

        exts = extensions or self.EXTENSIONS

        self.files_a = sorted(
            f for f in os.listdir(self.root_a) if Path(f).suffix.lower() in exts
        )
        self.files_b = sorted(
            f for f in os.listdir(self.root_b) if Path(f).suffix.lower() in exts
        )

        if len(self.files_a) == 0:
            raise FileNotFoundError(f"No images found in {self.root_a}")
        if len(self.files_b) == 0:
            raise FileNotFoundError(f"No images found in {self.root_b}")

    def __len__(self) -> int:
        return max(len(self.files_a), len(self.files_b))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        img_a = Image.open(
            self.root_a / self.files_a[index % len(self.files_a)]
        ).convert("RGB")
        # Random index for domain B to avoid fixed pairings
        img_b = Image.open(
            self.root_b / self.files_b[random.randint(0, len(self.files_b) - 1)]
        ).convert("RGB")

        if self.transform_a is not None:
            img_a = self.transform_a(img_a)
        if self.transform_b is not None:
            img_b = self.transform_b(img_b)

        return {"A": img_a, "B": img_b}
