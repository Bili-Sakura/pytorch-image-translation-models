# Credits: Built on open-source libraries and papers acknowledged in README.md citations.
"""Example scripts for training and inference with image translation models.

Subpackages
-----------
``community/``
    Community-contributed pipelines, each in a dedicated subfolder with
    ``model.py``, ``pipeline.py``, ``train.py``, and ``README.md``.
``i2sb/``
    I2SB (Image-to-Image Schrödinger Bridge) task configs and training loop.
``inference/``
    Unified inference script supporting all bridge-diffusion methods.
``pix2pix/``
    Pix2Pix paired image-to-image translation training.
``stegogan/``
    StegoGAN non-bijective unpaired image translation training.
``stargan/``
    StarGAN multi-domain image translation training.
``cut/``
    CUT contrastive unpaired translation training.

All training and inference examples import components from ``src/`` — no
duplicated model, scheduler, or pipeline code.
"""
