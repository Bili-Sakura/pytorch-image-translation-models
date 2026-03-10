"""StarGAN training utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models.stargan import StarGANDiscriminator, StarGANGenerator


@dataclass
class StarGANTrainingConfig:
    """Configuration for StarGAN adversarial training."""

    image_size: int = 128
    c_dim: int = 5
    g_conv_dim: int = 64
    d_conv_dim: int = 64
    g_repeat_num: int = 6
    d_repeat_num: int = 6
    lambda_cls: float = 1.0
    lambda_rec: float = 10.0
    lambda_gp: float = 10.0
    n_critic: int = 5
    g_lr: float = 1e-4
    d_lr: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    device: str = "cpu"


class StarGANTrainer:
    """Training harness with WGAN-GP + domain classification + reconstruction."""

    def __init__(self, config: StarGANTrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        self.G = StarGANGenerator(
            conv_dim=config.g_conv_dim,
            c_dim=config.c_dim,
            repeat_num=config.g_repeat_num,
        ).to(self.device)
        self.D = StarGANDiscriminator(
            image_size=config.image_size,
            conv_dim=config.d_conv_dim,
            c_dim=config.c_dim,
            repeat_num=config.d_repeat_num,
        ).to(self.device)

        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(),
            lr=config.g_lr,
            betas=(config.beta1, config.beta2),
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(),
            lr=config.d_lr,
            betas=(config.beta1, config.beta2),
        )

    @staticmethod
    def _classification_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BCE-with-logits over multi-hot domain attributes.
        return F.binary_cross_entropy_with_logits(logit, target, reduction="sum") / logit.size(0)

    def _gradient_penalty(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        weight = torch.ones_like(y, device=self.device)
        dydx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]
        dydx = dydx.view(dydx.size(0), -1)
        return torch.mean((torch.norm(dydx, dim=1) - 1) ** 2)

    def train_discriminator(
        self,
        real_x: torch.Tensor,
        real_labels: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> dict[str, float]:
        """One discriminator update."""
        real_x = real_x.to(self.device)
        real_labels = real_labels.to(self.device)
        target_labels = target_labels.to(self.device)

        out_src, out_cls = self.D(real_x)
        d_loss_real = -torch.mean(out_src)
        d_loss_cls = self._classification_loss(out_cls, real_labels)

        with torch.no_grad():
            fake_x = self.G(real_x, target_labels)
        out_src_fake, _ = self.D(fake_x.detach())
        d_loss_fake = torch.mean(out_src_fake)

        alpha = torch.rand(real_x.size(0), 1, 1, 1, device=self.device)
        x_hat = (alpha * real_x.data + (1 - alpha) * fake_x.data).requires_grad_(True)
        out_src_hat, _ = self.D(x_hat)
        d_loss_gp = self._gradient_penalty(out_src_hat, x_hat)

        d_loss = (
            d_loss_real
            + d_loss_fake
            + self.config.lambda_cls * d_loss_cls
            + self.config.lambda_gp * d_loss_gp
        )

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        return {
            "d_loss": float(d_loss.item()),
            "d_real": float(d_loss_real.item()),
            "d_fake": float(d_loss_fake.item()),
            "d_cls": float(d_loss_cls.item()),
            "d_gp": float(d_loss_gp.item()),
        }

    def train_generator(
        self,
        real_x: torch.Tensor,
        real_labels: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> dict[str, float]:
        """One generator update."""
        real_x = real_x.to(self.device)
        real_labels = real_labels.to(self.device)
        target_labels = target_labels.to(self.device)

        fake_x = self.G(real_x, target_labels)
        out_src, out_cls = self.D(fake_x)
        g_loss_fake = -torch.mean(out_src)
        g_loss_cls = self._classification_loss(out_cls, target_labels)

        rec_x = self.G(fake_x, real_labels)
        g_loss_rec = torch.mean(torch.abs(real_x - rec_x))

        g_loss = g_loss_fake + self.config.lambda_rec * g_loss_rec + self.config.lambda_cls * g_loss_cls

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return {
            "g_loss": float(g_loss.item()),
            "g_fake": float(g_loss_fake.item()),
            "g_cls": float(g_loss_cls.item()),
            "g_rec": float(g_loss_rec.item()),
        }

    def train_step(
        self,
        real_x: torch.Tensor,
        real_labels: torch.Tensor,
        target_labels: torch.Tensor,
        step: int,
    ) -> dict[str, float]:
        """Run one full StarGAN optimization step."""
        logs = self.train_discriminator(real_x, real_labels, target_labels)
        if step % self.config.n_critic == 0:
            logs.update(self.train_generator(real_x, real_labels, target_labels))
        return logs

