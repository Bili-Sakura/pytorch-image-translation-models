# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Community-contributed pipelines for image translation models.

Community pipelines are self-contained, single-file modules contributed by the
community.  They follow the same pattern as
`Hugging Face diffusers <https://github.com/huggingface/diffusers/tree/main/examples/community>`_:
each file bundles *all* model, loss, and utility code needed for training or
inference so that the pipeline works without any other project code.

**How to add a new community pipeline**

1. Create a single Python file under ``examples/community/`` named after the
   model, e.g. ``my_model.py``.
2. Put all network definitions, loss helpers, and a simple trainer / inference
   class inside that file.
3. Add a short docstring at the top describing the paper, usage, and citation.
4. Add an entry in ``examples/community/README.md``.

Available community pipelines:

* ``parallel_gan.py`` – Parallel-GAN (Wang et al., TGRS 2022)
* ``e3diff.py`` – E3Diff (Qin et al., IEEE GRSL 2024)
"""
