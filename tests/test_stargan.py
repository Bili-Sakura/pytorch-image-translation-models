# Credits: StarGAN from Choi et al. "Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation" CVPR 2018.

"""Tests for native StarGAN integration."""

import torch


def test_stargan_trainer_discriminator_step():
    from examples.stargan.train_stargan import StarGANTrainer, StarGANTrainingConfig

    cfg = StarGANTrainingConfig(
        image_size=32,
        c_dim=5,
        g_conv_dim=16,
        d_conv_dim=16,
        g_repeat_num=2,
        d_repeat_num=3,
        n_critic=2,
        device="cpu",
    )
    trainer = StarGANTrainer(cfg)
    real_x = torch.randn(2, 3, 32, 32)
    real_labels = torch.randint(0, 2, (2, 5)).float()
    target_labels = torch.randint(0, 2, (2, 5)).float()

    logs = trainer.train_discriminator(real_x, real_labels, target_labels)
    assert "d_loss" in logs
    assert "d_gp" in logs


def test_stargan_trainer_generator_step():
    from examples.stargan.train_stargan import StarGANTrainer, StarGANTrainingConfig

    cfg = StarGANTrainingConfig(
        image_size=32,
        c_dim=5,
        g_conv_dim=16,
        d_conv_dim=16,
        g_repeat_num=2,
        d_repeat_num=3,
        n_critic=2,
        device="cpu",
    )
    trainer = StarGANTrainer(cfg)
    real_x = torch.randn(2, 3, 32, 32)
    real_labels = torch.randint(0, 2, (2, 5)).float()
    target_labels = torch.randint(0, 2, (2, 5)).float()

    logs = trainer.train_generator(real_x, real_labels, target_labels)
    assert "g_loss" in logs
    assert "g_rec" in logs

