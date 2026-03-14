# Copyright (c) 2026 EarthBridge Team.
# Credits: SynDiff (Özbey et al., IEEE TMI 2023) - https://github.com/icon-lab/SynDiff

"""SynDiff training entry point.

Training is performed by the original SynDiff repository. This module provides
a convenience wrapper to run it with the project's layout.

Usage with original SynDiff:
    cd /path/to/SynDiff
    python train.py --image_size 256 --exp exp_syndiff --num_channels 2 ...

See README.md for full training command and checkpoint layout.
"""
