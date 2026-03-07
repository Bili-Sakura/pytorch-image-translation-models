# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Community-contributed pipelines for image translation models.

Each community pipeline lives in its own subfolder under
``examples/community/<model_name>/`` and contains:

* ``model.py`` – Network architecture definitions.
* ``pipeline.py`` – Configuration and training / inference pipeline.
* ``readme.md`` – Documentation, usage examples, and citation.

These pipelines are self-contained: they bundle all model, loss, and utility
code needed for training or inference so that they work without importing any
other project code from ``src/``.

**How to add a new community pipeline**

1. Create a new subfolder under ``examples/community/`` named after the model,
   e.g. ``examples/community/my_model/``.
2. Add ``model.py`` with all network definitions and loss helpers.
3. Add ``pipeline.py`` with a configuration dataclass and a trainer / inference
   class.
4. Add ``readme.md`` describing the paper, usage, and citation.
5. Add an ``__init__.py`` that re-exports public symbols.
6. Add an entry in ``examples/community/README.md``.
7. Add tests in ``tests/test_community_pipelines.py``.

Available community pipelines:

* ``parallel_gan/`` – Parallel-GAN (Wang et al., TGRS 2022)
* ``e3diff/`` – E3Diff (Qin et al., IEEE GRSL 2024)
"""
