# Copyright (c) 2026 EarthBridge Team.
# Credits: AlignFlow (Grover et al., AAAI 2020) https://github.com/ermongroup/alignflow

import torch
import torch.nn as nn
from itertools import chain

from .. import util
from ..patch_gan import PatchGAN
from ..real_nvp import RealNVP, RealNVPLoss


class Flow2Flow(nn.Module):
    """Flow2Flow Model – normalizing flows for unpaired image-to-image translation.

    Uses two RealNVP models (g_src: X <-> Z, g_tgt: Y <-> Z) mapping to a shared
    latent space, with hybrid GAN-MLE objective.
    """
    def __init__(self, args):
        super(Flow2Flow, self).__init__()
        self.device = 'cuda' if len(args.gpu_ids) > 0 else 'cpu'
        self.gpu_ids = args.gpu_ids
        self.is_training = args.is_training

        self.in_channels = args.num_channels
        self.out_channels = 4 ** (args.num_scales - 1) * self.in_channels

        self.g_src = RealNVP(num_scales=args.num_scales,
                             in_channels=args.num_channels,
                             mid_channels=args.num_channels_g,
                             num_blocks=args.num_blocks,
                             un_normalize_x=True,
                             no_latent=False)
        util.init_model(self.g_src, init_method=args.initializer)
        self.g_tgt = RealNVP(num_scales=args.num_scales,
                            in_channels=args.num_channels,
                            mid_channels=args.num_channels_g,
                            num_blocks=args.num_blocks,
                            un_normalize_x=True,
                            no_latent=False)
        util.init_model(self.g_tgt, init_method=args.initializer)

        if self.is_training:
            self.d_tgt = PatchGAN(args)
            self.d_src = PatchGAN(args)
            self._data_parallel()
            self.max_grad_norm = args.clip_gradient
            self.lambda_mle = args.lambda_mle
            self.mle_loss_fn = RealNVPLoss()
            self.gan_loss_fn = util.GANLoss(device=self.device, use_least_squares=True)
            self.clamp_jacobian = args.clamp_jacobian
            self.jc_loss_fn = util.JacobianClampingLoss(args.jc_lambda_min, args.jc_lambda_max)
            g_src_params = util.get_param_groups(self.g_src, args.weight_norm_l2, norm_suffix='weight_g')
            g_tgt_params = util.get_param_groups(self.g_tgt, args.weight_norm_l2, norm_suffix='weight_g')
            self.opt_g = torch.optim.Adam(chain(g_src_params, g_tgt_params),
                                          lr=args.rnvp_lr,
                                          betas=(args.rnvp_beta_1, args.rnvp_beta_2))
            self.opt_d = torch.optim.Adam(chain(self.d_tgt.parameters(), self.d_src.parameters()),
                                         lr=args.lr, betas=(args.beta_1, args.beta_2))
            self.optimizers = [self.opt_g, self.opt_d]
            self.schedulers = [util.get_lr_scheduler(opt, args) for opt in self.optimizers]
            buffer_capacity = 50 if args.use_mixer else 0
            self.src2tgt_buffer = util.ImageBuffer(buffer_capacity)
            self.tgt2src_buffer = util.ImageBuffer(buffer_capacity)
        else:
            self._data_parallel()

        self.src = self.src2lat = self.src2tgt = self.tgt = self.tgt2lat = self.tgt2src = None
        self.src_jc = self.tgt_jc = self.src2tgt_jc = self.tgt2src_jc = None
        self.loss_d_tgt = self.loss_d_src = self.loss_d = None
        self.loss_gan_src = self.loss_gan_tgt = self.loss_gan = None
        self.loss_mle_src = self.loss_mle_tgt = self.loss_mle = None
        self.loss_jc_src = self.loss_jc_tgt = self.loss_jc = self.loss_g = None

    def set_inputs(self, src_input, tgt_input=None):
        self.src = src_input.to(self.device)
        if tgt_input is not None:
            self.tgt = tgt_input.to(self.device)

    def forward(self):
        pass

    def test(self):
        with torch.no_grad():
            src2lat, _ = self.g_src(self.src, reverse=False)
            src2lat2tgt, _ = self.g_tgt(src2lat, reverse=True)
            self.src2tgt = torch.tanh(src2lat2tgt)
            tgt2lat, _ = self.g_tgt(self.tgt, reverse=False)
            tgt2lat2src, _ = self.g_src(tgt2lat, reverse=True)
            self.tgt2src = torch.tanh(tgt2lat2src)

    def _forward_d(self, d, real_img, fake_img):
        loss_real = self.gan_loss_fn(d(real_img), is_tgt_real=True)
        loss_fake = self.gan_loss_fn(d(fake_img.detach()), is_tgt_real=False)
        return 0.5 * (loss_real + loss_fake)

    def backward_d(self):
        src2tgt = self.src2tgt_buffer.sample(self.src2tgt)
        self.loss_d_tgt = self._forward_d(self.d_tgt, self.tgt, src2tgt)
        tgt2src = self.tgt2src_buffer.sample(self.tgt2src)
        self.loss_d_src = self._forward_d(self.d_src, self.src, tgt2src)
        self.loss_d = self.loss_d_tgt + self.loss_d_src
        self.loss_d.backward()

    def backward_g(self):
        if self.clamp_jacobian:
            self._jc_preprocess()
        self.src2lat, sldj_src2lat = self.g_src(self.src, reverse=False)
        self.loss_mle_src = self.lambda_mle * self.mle_loss_fn(self.src2lat, sldj_src2lat)
        src2tgt, _ = self.g_tgt(self.src2lat, reverse=True)
        self.src2tgt = torch.tanh(src2tgt)
        self.tgt2lat, sldj_tgt2lat = self.g_tgt(self.tgt, reverse=False)
        self.loss_mle_tgt = self.lambda_mle * self.mle_loss_fn(self.tgt2lat, sldj_tgt2lat)
        tgt2src, _ = self.g_src(self.tgt2lat, reverse=True)
        self.tgt2src = torch.tanh(tgt2src)
        if self.clamp_jacobian:
            self._jc_postprocess()
            self.loss_jc_src = self.jc_loss_fn(self.src2tgt, self.src2tgt_jc, self.src, self.src_jc)
            self.loss_jc_tgt = self.jc_loss_fn(self.tgt2src, self.tgt2src_jc, self.tgt, self.tgt_jc)
            self.loss_jc = self.loss_jc_src + self.loss_jc_tgt
        else:
            self.loss_jc_src = self.loss_jc_tgt = self.loss_jc = 0.
        self.loss_gan_src = self.gan_loss_fn(self.d_tgt(self.src2tgt), is_tgt_real=True)
        self.loss_gan_tgt = self.gan_loss_fn(self.d_src(self.tgt2src), is_tgt_real=True)
        self.loss_gan = self.loss_gan_src + self.loss_gan_tgt
        self.loss_mle = self.loss_mle_src + self.loss_mle_tgt
        self.loss_g = self.loss_gan + self.loss_mle + self.loss_jc
        self.loss_g.backward()

    def train_iter(self):
        self.forward()
        self.opt_g.zero_grad()
        self.backward_g()
        util.clip_grad_norm(self.opt_g, self.max_grad_norm)
        self.opt_g.step()
        self.opt_d.zero_grad()
        self.backward_d()
        util.clip_grad_norm(self.opt_d, self.max_grad_norm)
        self.opt_d.step()

    def get_loss_dict(self):
        loss_dict = {'loss_gan': self.loss_gan, 'loss_jc': self.loss_jc, 'loss_mle': self.loss_mle,
                     'loss_g': self.loss_g, 'loss_d_src': self.loss_d_src,
                     'loss_d_tgt': self.loss_d_tgt, 'loss_d': self.loss_d}
        return {k: v.item() for k, v in loss_dict.items() if isinstance(v, torch.Tensor)}

    def get_image_dict(self):
        image_tensor_dict = {'src': self.src, 'src2tgt': self.src2tgt}
        if self.is_training:
            image_tensor_dict.update({'tgt': self.tgt, 'tgt2src': self.tgt2src})
        return {k: util.un_normalize(v) for k, v in image_tensor_dict.items()}

    def on_epoch_end(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_learning_rate(self):
        return self.optimizers[0].param_groups[0]['lr']

    def _data_parallel(self):
        self.g_src = nn.DataParallel(self.g_src, self.gpu_ids).to(self.device)
        self.g_tgt = nn.DataParallel(self.g_tgt, self.gpu_ids).to(self.device)
        if self.is_training:
            self.d_src = nn.DataParallel(self.d_src, self.gpu_ids).to(self.device)
            self.d_tgt = nn.DataParallel(self.d_tgt, self.gpu_ids).to(self.device)

    def _jc_preprocess(self):
        delta = torch.randn_like(self.src)
        src_jc = self.src + delta / delta.norm()
        src_jc.clamp_(-1, 1)
        self.src = torch.cat((self.src, src_jc), dim=0)
        delta = torch.randn_like(self.tgt)
        tgt_jc = self.tgt + delta / delta.norm()
        tgt_jc.clamp_(-1, 1)
        self.tgt = torch.cat((self.tgt, tgt_jc), dim=0)

    def _jc_postprocess(self):
        self.src, self.src_jc = self.src.chunk(2, dim=0)
        self.tgt, self.tgt_jc = self.tgt.chunk(2, dim=0)
        self.src2tgt, self.src2tgt_jc = self.src2tgt.chunk(2, dim=0)
        self.tgt2src, self.tgt2src_jc = self.tgt2src.chunk(2, dim=0)
