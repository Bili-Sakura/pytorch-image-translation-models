# Copyright (c) 2026 EarthBridge Team.
# Credits: AlignFlow (Grover et al., AAAI 2020) https://github.com/ermongroup/alignflow

"""Shell / misc utilities."""

from argparse import ArgumentTypeError


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def args_to_list(csv, allow_empty, arg_type=int, allow_negative=True):
    """Convert comma-separated arguments to a list."""
    arg_vals = [arg_type(d) for d in str(csv).split(",")]
    if not allow_negative:
        arg_vals = [v for v in arg_vals if v >= 0]
    if not allow_empty and len(arg_vals) == 0:
        return None
    return arg_vals


def str_to_bool(v):
    """Convert an argument string into its boolean value."""
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")
