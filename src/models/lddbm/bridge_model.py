# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

import torch
import torch.nn

from src.models.lddbm.diffusion_utils import vp_logs, vp_logsnr
from src.models.lddbm.karras_diffusion import KarrasDenoiser, karras_sample
from src.models.lddbm.resample import ScheduleSampler
from src.models.lddbm.nn_utils import append_dims


class BridgeModel(torch.nn.Module):
    def __init__(
        self,
        denoiser: torch.nn.Module,
        diffusion: KarrasDenoiser,
        schedule_sampler: ScheduleSampler,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.diffusion = diffusion
        self.schedule_sampler = schedule_sampler

    def forward(self, z_0, z_T):
        t, weights = self.schedule_sampler.sample(z_0.shape[0], z_0.device)
        sigmas = torch.minimum(t, torch.ones_like(t) * self.diffusion.sigma_max)
        dims = z_0.ndim
        noise = torch.randn_like(z_0)

        x_t = self.bridge_sample(
            z_0,
            z_T,
            sigmas,
            noise,
            dims,
            self.diffusion.sigma_max,
            self.diffusion.beta_d,
            self.diffusion.beta_min,
        )
        _, denoised = self._denoise(
            x_t,
            sigmas,
            z_T,
            self.diffusion.pred_mode,
            self.diffusion.sigma_max,
            self.diffusion.beta_d,
            self.diffusion.beta_min,
            self.diffusion.c,
            self.diffusion.sigma_data_end,
            self.diffusion.sigma_data,
            self.diffusion.cov_xy,
        )

        return denoised

    def sample(self, xT, sampling_steps=40):
        sample, _ = karras_sample(
            self.diffusion,
            self,
            xT,
            sampling_steps,
            device=xT.device,
            sampler=self.diffusion.sampler,
            sigma_min=self.diffusion.sigma_min,
            sigma_max=self.diffusion.sigma_max,
            churn_step_ratio=self.diffusion.churn_step_ratio,
            rho=self.diffusion.rho,
            guidance=self.diffusion.guidance,
        )

        return sample

    def _denoise(
        self,
        x_t,
        sigmas,
        xT,
        pred_mode,
        sigma_max,
        beta_d,
        beta_min,
        c,
        sigma_data_end,
        sigma_data,
        cov_xy,
        aux_cond=None,
    ):
        def get_bridge_scalings(sigma):
            if pred_mode == "ve":
                A = (
                    sigma**4 / sigma_max**4 * sigma_data_end**2
                    + (1 - sigma**2 / sigma_max**2) ** 2 * sigma_data**2
                    + 2
                    * sigma**2
                    / sigma_max**2
                    * (1 - sigma**2 / sigma_max**2)
                    * cov_xy
                    + c**2 * sigma**2 * (1 - sigma**2 / sigma_max**2)
                )
                c_in = 1 / (A) ** 0.5
                c_skip = (
                    (1 - sigma**2 / sigma_max**2) * sigma_data**2
                    + sigma**2 / sigma_max**2 * cov_xy
                ) / A
                c_out = (
                    (sigma / sigma_max) ** 4
                    * (sigma_data_end**2 * sigma_data**2 - cov_xy**2)
                    + sigma_data**2 * c**2 * sigma**2 * (1 - sigma**2 / sigma_max**2)
                ) ** 0.5 * c_in
                return c_skip, c_out, c_in

            elif pred_mode == "vp":

                logsnr_t = vp_logsnr(sigma, beta_d, beta_min)
                logsnr_T = vp_logsnr(1, beta_d, beta_min)
                logs_t = vp_logs(sigma, beta_d, beta_min)
                logs_T = vp_logs(1, beta_d, beta_min)

                a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
                b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
                c_t = -torch.expm1(logsnr_T - logsnr_t) * (2 * logs_t - logsnr_t).exp()

                A = (
                    a_t**2 * sigma_data_end**2
                    + b_t**2 * sigma_data**2
                    + 2 * a_t * b_t * cov_xy
                    + c**2 * c_t
                )

                c_in = 1 / (A) ** 0.5
                c_skip = (b_t * sigma_data**2 + a_t * cov_xy) / A
                c_out = (
                    a_t**2 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2)
                    + sigma_data**2 * c**2 * c_t
                ) ** 0.5 * c_in
                return c_skip, c_out, c_in

        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in get_bridge_scalings(sigmas)
        ]

        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        model_output = self.denoiser(c_in * x_t, rescaled_t, xT)
        denoised = c_out * model_output + c_skip * x_t

        return model_output, denoised

    def bridge_sample(self, x0, xT, t, noise, dims, sigma_max, beta_d, beta_min):
        t = append_dims(t, dims)
        # std_t = th.sqrt(t)* th.sqrt(1 - t / self.sigma_max)
        if self.diffusion.pred_mode.startswith("ve"):
            std_t = t * torch.sqrt(1 - t**2 / sigma_max**2)
            mu_t = t**2 / sigma_max**2 * xT + (1 - t**2 / sigma_max**2) * x0
            samples = mu_t + std_t * noise
        elif self.diffusion.pred_mode.startswith("vp"):
            logsnr_t = vp_logsnr(t, beta_d, beta_min)
            logsnr_T = vp_logsnr(sigma_max, beta_d, beta_min)
            logs_t = vp_logs(t, beta_d, beta_min)
            logs_T = vp_logs(sigma_max, beta_d, beta_min)

            a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
            b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (
                logs_t - logsnr_t / 2
            ).exp()

            samples = a_t * xT + b_t * x0 + std_t * noise

        return samples

    def convert_to_fp16(self):
        self.denoiser.convert_to_fp16()
