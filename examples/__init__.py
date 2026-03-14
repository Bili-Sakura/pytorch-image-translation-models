"""Example scripts for training and inference with image translation models.

Subpackages
-----------
``community/``
    Community-contributed pipelines, each in a dedicated subfolder with
    ``model.py``, ``pipeline.py``, ``train.py``, and ``README.md``.

Paired image-to-image (src pipelines)
------------------------------------
``bbdm/``      BBDM (Brownian Bridge Diffusion) training.
``bdbm/``      BDBM (Bidirectional Diffusion Bridge) training.
``bibbdm/``    BiBBDM bidirectional bridge training.
``cdtsde/``    CDTSDE (Adaptive Domain Shift) training.
``dbim/``      DBIM (Diffusion Bridge Implicit) training.
``ddbm/``      DDBM (Denoising Diffusion Bridge) training.
``ddib/``      DDIB (Dual Diffusion Implicit Bridges) – two UNets.
``dbim/``      DBIM (Diffusion Bridge Implicit Models) training.
``ecsi/``, ``fcdm/``
    See community examples for full training workflows.
``lddbm/``     LDDBM (Latent Diffusion Bridge Model) super-resolution 16→128 training.
``i2sb/``      I2SB (Image-to-Image Schrödinger Bridge) – run: ``python -m examples.i2sb``.
``lbm/``       LBM (Latent Bridge Matching) training.
``local_diffusion/``   Local Diffusion training (ECCV 2024).
``pix2pix/``   Pix2Pix paired training.
``pix2pixhd/`` Pix2PixHD high-resolution paired training.
``stargan/``   StarGAN multi-domain training.

Unpaired image-to-image
-----------------------
``cut/``       CUT contrastive unpaired translation.
``stegogan/``  StegoGAN non-bijective unpaired translation.
``unsb/``      UNSB (Unpaired Neural Schrödinger Bridge).

All training examples import components from ``src/`` — no duplicated model,
scheduler, or pipeline code.
"""
