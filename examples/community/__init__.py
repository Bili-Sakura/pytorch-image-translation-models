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

* ``ddbm/`` – DDBM for OpenAI-style checkpoints (Zhou et al., ICLR 2024)
* ``bbdm/`` – BBDM for OpenAI-style checkpoints (Li et al., CVPR 2023)
* ``parallel_gan/`` – Parallel-GAN (Wang et al., TGRS 2022)
* ``e3diff/`` – E3Diff (Qin et al., IEEE GRSL 2024)
* ``openearthmap_sar`` – OpenEarthMap-SAR CUT models for SAR ↔ optical translation (Park et al., ECCV 2020)
* ``sar2optical`` – Pix2Pix cGAN SAR → optical translation (Isola et al., CVPR 2017; yuuIind/SAR2Optical)
* ``diffuseit/`` – DiffuseIT (Kwon & Ye, ICLR 2023)
* ``diffusionrouter/`` – DiffusionRouter (kvmduc) universal multi-domain routing with conditional diffusion
* ``syndiff/`` – SynDiff (Özbey et al., IEEE TMI 2023) – unsupervised medical image translation with adversarial diffusion
* ``selfrdb/`` – SelfRDB (Arslan et al., Medical Image Analysis 2024) – self-consistent recursive diffusion bridge for medical synthesis
* ``sdedit/`` – SDEdit (Meng et al., ICLR 2022) – guided image synthesis and editing with SDEs; requires local [ermongroup/SDEdit](https://github.com/ermongroup/SDEdit) checkout
* ``hneg_src/`` – Hneg-SRC (Jung et al., CVPR 2022) – patch-wise semantic relation contrastive learning for unpaired translation
* ``negcut/`` – NEGCUT (Wang et al., ICCV 2021) – adversarial hard-negative generation for contrastive unpaired translation
* ``flsesim/`` – F-LSeSim (Zheng et al., CVPR 2021) – spatially-correlative loss for structure-preserving unpaired translation
* ``cyclegan_turbo/`` – CycleGAN-Turbo (Parmar et al., 2024) – one-step unpaired SD-Turbo translation
* ``pix2pix_turbo/`` – pix2pix-turbo (Parmar et al., 2024) – one-step paired SD-Turbo translation
* ``cyclegan/`` – CycleGAN (Zhu et al., ICCV 2017) – classic unpaired translation from junyanz/pytorch-CycleGAN-and-pix2pix
* ``pix2pix/`` – pix2pix (Isola et al., CVPR 2017) – classic paired translation from junyanz/pytorch-CycleGAN-and-pix2pix
"""
