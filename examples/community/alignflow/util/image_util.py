# Copyright (c) 2026 EarthBridge Team.
# Credits: AlignFlow (Grover et al., AAAI 2020) https://github.com/ermongroup/alignflow

"""Image utilities."""

import math
import torch


def un_normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """Reverse normalization of an image. Move to CPU."""
    tensor = tensor.cpu().float()
    for i in range(len(mean)):
        tensor[:, i, :, :] *= std[i]
        tensor[:, i, :, :] += mean[i]
    tensor *= 255.0
    tensor = tensor.type(torch.uint8)
    return tensor


def make_grid(tensor, nrow=8, padding=2, normalize=False, range_=None, scale_each=False, pad_value=0):
    """Make a grid of images."""
    if not (
        torch.is_tensor(tensor)
        or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))
    ):
        raise TypeError("tensor or list of tensors expected, got {}".format(type(tensor)))
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)
    if tensor.dim() == 2:
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:
        if tensor.size(0) == 1:
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))
    if tensor.dim() == 4 and tensor.size(1) == 1:
        tensor = torch.cat((tensor, tensor, tensor), 1)
    if normalize is True:
        tensor = tensor.clone()
        if range_ is not None:
            assert isinstance(range_, tuple)

        def norm_ip(img, min_val, max_val):
            img.clamp_(min=min_val, max=max_val)
            img.add_(-min_val).div_(max_val - min_val + 1e-5)

        def norm_range(t, rng):
            if rng is not None:
                norm_ip(t, rng[0], rng[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:
                norm_range(t, range_)
        else:
            norm_range(tensor, range_)
    if tensor.size(0) == 1:
        return tensor.squeeze()
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding)
    if isinstance(pad_value, (float, int)):
        grid.fill_(pad_value)
    else:
        if len(pad_value) != 3:
            raise ValueError("pad_value per channel must have 3 elements")
        for i, v in enumerate(pad_value):
            grid[i, :, :] = v
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding).narrow(
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k += 1
    return grid
