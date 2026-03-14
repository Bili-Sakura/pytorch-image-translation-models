# Copyright (c) 2026 EarthBridge Team.
# Credits: AlignFlow (Grover et al., AAAI 2020) https://github.com/ermongroup/alignflow

"""Optimization utilities."""

from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler as torch_scheduler


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """Clip the norm of the gradients for all parameters."""
    if max_norm > 0:
        for group in optimizer.param_groups:
            clip_grad_norm_(group["params"], max_norm, norm_type)


def get_lr_scheduler(optimizer, args):
    """Get learning rate scheduler."""
    if args.lr_policy == "step":
        return torch_scheduler.StepLR(optimizer, step_size=args.lr_step_epochs, gamma=0.1)
    elif args.lr_policy == "plateau":
        return torch_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif args.lr_policy == "linear":
        def get_lr_multiplier(epoch):
            init_epoch = 1
            return 1.0 - max(
                0, epoch + init_epoch - args.lr_warmup_epochs
            ) / float(args.lr_decay_epochs + 1)

        return torch_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
    else:
        raise NotImplementedError("Invalid learning rate policy: {}".format(args.lr_policy))
