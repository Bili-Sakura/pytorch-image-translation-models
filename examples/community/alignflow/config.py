# Copyright (c) 2026 EarthBridge Team.
# Credits: AlignFlow (Grover et al., AAAI 2020) - https://github.com/ermongroup/alignflow

"""Configuration for AlignFlow models: CycleFlow, Flow2Flow, CycleGAN."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AlignFlowConfig:
    """Configuration for AlignFlow models.

    Supports CycleFlow, Flow2Flow, and CycleGAN for unpaired image-to-image translation.
    """

    model: str = "CycleFlow"  # CycleFlow | Flow2Flow | CycleGAN
    num_channels: int = 3
    num_channels_d: int = 64
    num_channels_g: int = 64
    num_scales: int = 2
    num_blocks: int = 4
    kernel_size_d: int = 4
    norm_type: str = "instance"
    initializer: str = "normal"
    use_mixer: bool = True
    use_dropout: bool = False
    clamp_jacobian: bool = False
    jc_lambda_min: float = 1.0
    jc_lambda_max: float = 20.0
    lambda_mle: float = 1e-4  # Flow2Flow only
    lambda_src: float = 10.0  # CycleGAN only
    lambda_tgt: float = 10.0
    lambda_id: float = 0.5
    lr: float = 2e-4
    rnvp_lr: float = 2e-4
    beta_1: float = 0.5
    beta_2: float = 0.999
    rnvp_beta_1: float = 0.5
    rnvp_beta_2: float = 0.999
    weight_norm_l2: float = 5e-5
    clip_gradient: float = 0.0
    lr_policy: str = "linear"
    lr_step_epochs: int = 100
    lr_warmup_epochs: int = 100
    lr_decay_epochs: int = 100
    device: str = "cuda"
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    is_training: bool = True
    save_dir: str = "checkpoints/alignflow"
    batch_size: int = 16
    epochs: int = 200
    resolution: int = 128
    log_every: int = 100
    save_every: int = 10

    def to_args(self) -> "AlignFlowConfig":
        """Return self as args-like object (this config already has all attributes)."""
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "AlignFlowConfig":
        """Create config from dict (e.g. loaded from JSON)."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)
