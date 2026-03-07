# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Community-contributed pipelines for image translation models.

Each community pipeline lives in its own subfolder under
``examples/community/<model_name>/`` and contains:

* ``model.py``    – Network architectures and losses.
* ``pipeline.py`` – Inference / pipeline logic.
* ``train.py``    – Training configuration and harness.
* ``README.md``   – Usage examples and citation information.

**How to add a new community pipeline**

1. Create a subfolder under ``examples/community/`` named after the model,
   e.g. ``examples/community/my_model/``.
2. Add ``model.py``, ``pipeline.py``, ``train.py``, and ``README.md``.
3. Create an ``__init__.py`` that re-exports all public symbols.
4. Add an entry in ``examples/community/README.md``.

Available community pipelines:

* ``parallel_gan/`` – Parallel-GAN (Wang et al., TGRS 2022)
* ``e3diff/`` – E3Diff (Qin et al., IEEE GRSL 2024)
"""
