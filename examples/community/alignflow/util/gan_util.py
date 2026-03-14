# Copyright (c) 2026 EarthBridge Team.
# Credits: AlignFlow (Grover et al., AAAI 2020) https://github.com/ermongroup/alignflow

"""GAN-related utilities."""

import random
import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """Module for computing the GAN loss for the generator."""

    def __init__(self, device, use_least_squares=False):
        super().__init__()
        self.loss_fn = nn.MSELoss() if use_least_squares else nn.BCELoss()
        self.real_label = None
        self.fake_label = None
        self.device = device

    def _get_label_tensor(self, input_, is_tgt_real):
        if is_tgt_real and (self.real_label is None or self.real_label.numel() != input_.numel()):
            self.real_label = torch.ones_like(input_, device=self.device, requires_grad=False)
        elif not is_tgt_real and (self.fake_label is None or self.fake_label.numel() != input_.numel()):
            self.fake_label = torch.zeros_like(input_, device=self.device, requires_grad=False)
        return self.real_label if is_tgt_real else self.fake_label

    def __call__(self, input_, is_tgt_real):
        label = self._get_label_tensor(input_, is_tgt_real)
        return self.loss_fn(input_, label)


class ImageBuffer:
    """Holds a buffer of old generated images for training."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def sample(self, images):
        if self.capacity == 0:
            return images
        mixed_images = []
        for new_img in images:
            new_img = torch.unsqueeze(new_img.data, 0)
            if len(self.buffer) < self.capacity:
                self.buffer.append(new_img)
                mixed_images.append(new_img)
            else:
                if random.uniform(0, 1) < 0.5:
                    mixed_images.append(new_img)
                else:
                    pool_img_idx = random.randint(0, len(self.buffer) - 1)
                    mixed_images.append(self.buffer[pool_img_idx].clone())
                    self.buffer[pool_img_idx] = new_img
        return torch.cat(mixed_images, 0)


class JacobianClampingLoss(nn.Module):
    """Jacobian Clamping loss for flow models."""

    def __init__(self, lambda_min=1.0, lambda_max=20.0):
        super().__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def forward(self, gz, gz_prime, z, z_prime):
        q = (gz - gz_prime).norm() / (z - z_prime).norm()
        l_max = (q.clamp(self.lambda_max, float("inf")) - self.lambda_max) ** 2
        l_min = (q.clamp(float("-inf"), self.lambda_min) - self.lambda_min) ** 2
        return l_max + l_min
