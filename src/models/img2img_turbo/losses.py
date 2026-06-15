# Copyright (c) 2026 EarthBridge Team.
# Credits: img2img-turbo from Parmar et al. 2024 —
# https://github.com/GaParmar/img2img-turbo

"""Optional DINO structure loss used during CycleGAN-Turbo training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision


def attn_cosine_sim(x: torch.Tensor, eps: float = 1e-08) -> torch.Tensor:
    x = x[0]
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    return (x @ x.permute(0, 2, 1)) / factor


class VitExtractor:
    BLOCK_KEY = "block"
    ATTN_KEY = "attn"
    PATCH_IMD_KEY = "patch_imd"
    QKV_KEY = "qkv"
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name: str, device: str | torch.device) -> None:
        self.model = torch.hub.load("facebookresearch/dino:main", model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers: list = []
        self.layers_dict = {key: [] for key in self.KEY_LIST}
        self.outputs_dict = {key: [] for key in self.KEY_LIST}
        self._init_hooks_data()

    def _init_hooks_data(self) -> None:
        for key in self.KEY_LIST:
            self.layers_dict[key] = list(range(12))
            self.outputs_dict[key] = []

    def _register_hooks(self) -> None:
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[self.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._block_hook()))
            if block_idx in self.layers_dict[self.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._attn_hook()))
            if block_idx in self.layers_dict[self.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._qkv_hook()))
            if block_idx in self.layers_dict[self.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._patch_hook()))

    def _clear_hooks(self) -> None:
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _block_hook(self):
        def hook(_model, _input, output):
            self.outputs_dict[self.BLOCK_KEY].append(output)
        return hook

    def _attn_hook(self):
        def hook(_model, _inp, output):
            self.outputs_dict[self.ATTN_KEY].append(output)
        return hook

    def _qkv_hook(self):
        def hook(_model, _inp, output):
            self.outputs_dict[self.QKV_KEY].append(output)
        return hook

    def _patch_hook(self):
        def hook(_model, _inp, output):
            self.outputs_dict[self.PATCH_IMD_KEY].append(output[0])
        return hook

    def get_patch_num(self, input_img_shape) -> int:
        _, _, h, w = input_img_shape
        patch_size = 8 if "8" in self.model_name else 16
        return 1 + (h // patch_size) * (w // patch_size)

    def get_head_num(self) -> int:
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self) -> int:
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        return qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[self.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_keys_from_input(self, input_img, layer_num: int):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        return self.get_keys_from_qkv(qkv_features, input_img.shape)

    def get_keys_self_sim_from_input(self, input_img, layer_num: int):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated = keys.transpose(0, 1).reshape(t, h * d)
        return attn_cosine_sim(concatenated[None, None, ...])


class DinoStructureLoss:
    """DINO self-similarity structure loss for validation during CycleGAN-Turbo training."""

    def __init__(self, device: str | torch.device = "cuda") -> None:
        self.extractor = VitExtractor(model_name="dino_vitb8", device=device)
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def calculate_global_ssim_loss(self, outputs, inputs) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(iter(outputs)).device if hasattr(outputs, "__iter__") else "cpu")
        for a, b in zip(inputs, outputs):
            with torch.no_grad():
                target = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss = loss + F.mse_loss(keys, target)
        return loss


__all__ = ["DinoStructureLoss", "VitExtractor"]
