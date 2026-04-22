import sys
from functools import partial
from inspect import isfunction
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from src.srda.utils.respacing import space_timesteps
from torch import nn

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
    if schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: Callable,
        image_size_x: int,
        image_size_y: int,
        channels: int = 3,
        loss_type: str = 'l1',
        conditional: bool = True,
        schedule_opt: Optional[None] = None,
        sample_image_num: int = 10,
        eta: float = 1
    ):
        super().__init__()
        self.channels = channels
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.sample_image_num = sample_image_num
        self.eta = eta
        self.supports_obs_guidance = True
        self._gaussian_kernel_cache = {}

    def set_loss(self, device):
        if self.loss_type == 'l1':
            # This is the mean absolute error (MAE)
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        elif self.loss_type == 'l3':
            # This is the root mean squared mean
            self.loss_func = torch.sqrt(nn.MSELoss(reduction='sum')).to(device)
        else:
            raise NotImplementedError()

    def _set_beta_and_alpfa(self, betas: np.array, device):
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = self.eta**2 * betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def set_new_noise_schedule(self, schedule_opt, device):

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])

        self._set_beta_and_alpfa(betas=betas, device=device)

    def set_noise_schedule_for_respacing(self, timestep_respacing: list[int], device):
        self.use_timesteps = space_timesteps(self.num_timesteps, timestep_respacing)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                # self.timestep_map.append(i)
        betas = torch.tensor(new_betas)
        self._set_beta_and_alpfa(betas, device)

    def _predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        obs_guidance=None,
        step_count: int = 0,
        clip_denoised=True,
        condition_x=None,
    ):

        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]
        ).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            noise_pred = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
        else:
            noise_pred = self.denoise_fn(x, noise_level)
        predicted_x0 = self._predict_start_from_noise(x_t=x, t=t, noise=noise_pred)

        predicted_x0, applied_guidance = self._apply_obs_guidance(
            predicted_x0, obs_guidance, step_count
        )
        if (
            applied_guidance
            and obs_guidance is not None
            and obs_guidance.get("recompute_eps", False)
        ):
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(1, 1, 1, 1)
            sqrt_one_minus_t = self.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1, 1)
            noise_pred = (x - sqrt_alpha_t * predicted_x0) / (sqrt_one_minus_t + 1e-12)

        first_term = self.sqrt_alphas_cumprod_prev[t] * predicted_x0
        model_variance = self.posterior_variance[t]
        second_term = torch.sqrt(1 - self.alphas_cumprod_prev[t] - model_variance) * noise_pred
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        third_term = torch.sqrt(model_variance) * noise
        return first_term + second_term + third_term

    @torch.no_grad()
    def p_sample_loop(
        self,
        x_in,
        continous=False,
        hide_progress_bar=False,
        obs_guidance=None,
    ):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//self.sample_image_num))
        runtime_guidance = None
        if not self.conditional:
            shape = x_in
            b = shape[0]
            if obs_guidance is not None:
                runtime_guidance = self._prepare_runtime_obs_guidance(
                    obs_guidance, device, b
                )
                runtime_guidance["num_steps_total"] = self.num_timesteps
            img = torch.randn(shape, device=device)
            ret_img = img
            for step_count, i in enumerate(
                tqdm(
                    reversed(range(0, self.num_timesteps)),
                    desc='sampling loop time step',
                    total=self.num_timesteps,
                    disable=hide_progress_bar,
                )
            ):
                img = self.p_sample(
                    img, i, obs_guidance=runtime_guidance, step_count=step_count
                )
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            b, _, nx, ny = x_in.shape
            if obs_guidance is not None:
                runtime_guidance = self._prepare_runtime_obs_guidance(
                    obs_guidance, device, b
                )
                runtime_guidance["num_steps_total"] = self.num_timesteps
            shape = (b, self.channels, nx, ny)
            img = torch.randn(shape, device=device)
            ret_img = img
            for step_count, i in enumerate(
                tqdm(
                    reversed(range(0, self.num_timesteps)),
                    desc='sampling loop time step',
                    total=self.num_timesteps,
                    disable=hide_progress_bar,
                )
            ):
                img = self.p_sample(
                    img,
                    i,
                    obs_guidance=runtime_guidance,
                    step_count=step_count,
                    condition_x=x_in,
                )
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-b:]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, self.image_size_x, self.image_size_y), continous)

    @torch.no_grad()
    def super_resolution(
        self, x_in, continous=False, hide_progress_bar=False, obs_guidance=None
    ):
        return self.p_sample_loop(
            x_in, continous, hide_progress_bar, obs_guidance=obs_guidance
        )

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in["SR"], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
        loss = self.loss_func(noise, x_recon)
        return loss

    @torch.no_grad()
    def calc_loss(self, x_in, noise=None):
        return self.p_losses(x_in, noise)

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

    def _prepare_runtime_obs_guidance(self, cfg, device, batch_size):
        if cfg is None:
            return None
        mask = cfg["mask"].to(device)
        target = cfg["target"].to(device)
        if mask.shape[0] != batch_size:
            mask = mask.expand(batch_size, -1, -1, -1)
            target = target.expand(batch_size, -1, -1, -1)
        return {
            "mode": cfg["mode"],
            "mask": mask,
            "target": target,
            "gamma": cfg["gamma"],
            "sigma": cfg["sigma"],
            "apply_every": cfg["apply_every"],
            "blur_sigma_px": cfg.get("blur_sigma_px", 0.0),
            "blur_sigma_px_final": cfg.get("blur_sigma_px_final", None),
            "blur_schedule_power": cfg.get("blur_schedule_power", 1.0),
            "tighten_final_steps": cfg.get("tighten_final_steps", 0),
            "recompute_eps": cfg.get("recompute_eps", False),
            "space": cfg.get("space", "pixel"),
        }

    def _apply_obs_guidance(self, predicted_x0, obs_guidance, step_count):
        if obs_guidance is None:
            return predicted_x0, False

        total_steps = int(obs_guidance.get("num_steps_total") or 0)
        tighten_steps = int(obs_guidance.get("tighten_final_steps") or 0)
        is_tight_step = (
            total_steps > 0
            and tighten_steps > 0
            and step_count >= (total_steps - tighten_steps)
        )

        apply_every = int(obs_guidance.get("apply_every") or 1)
        should_apply = is_tight_step or apply_every <= 1 or (step_count % apply_every == 0)
        if not should_apply:
            return predicted_x0, False

        blur_sigma_start = float(obs_guidance.get("blur_sigma_px", 0.0) or 0.0)
        blur_sigma_final = obs_guidance.get("blur_sigma_px_final", None)
        blur_sigma = blur_sigma_start
        if blur_sigma_final is not None and total_steps > 1:
            power = float(obs_guidance.get("blur_schedule_power", 1.0) or 1.0)
            power = max(0.0, power)
            progress = step_count / float(total_steps - 1)
            w = progress ** power
            blur_sigma = blur_sigma_start * (1.0 - w) + float(blur_sigma_final) * w
        mask = obs_guidance["mask"]
        target = obs_guidance["target"]
        mask_float = mask.to(predicted_x0.dtype)
        residual = (target - predicted_x0) * mask_float
        if blur_sigma > 0:
            residual = self._gaussian_blur(residual, blur_sigma)

        effective_mode = obs_guidance["mode"]
        if is_tight_step and effective_mode == "soft":
            effective_mode = "hard"

        if effective_mode == "hard":
            updated = predicted_x0 + residual
            return torch.where(mask, target, updated), True

        alpha = obs_guidance["gamma"] / (
            obs_guidance["gamma"] + obs_guidance["sigma"] ** 2
        )
        return predicted_x0 + alpha * residual, True

    def _gaussian_blur(self, tensor: torch.Tensor, sigma_px: float) -> torch.Tensor:
        if sigma_px <= 0:
            return tensor
        radius = max(1, int(3 * sigma_px))
        if radius <= 0:
            return tensor
        key = (tensor.device, tensor.dtype, float(sigma_px), radius)
        kernel = self._gaussian_kernel_cache.get(key)
        if kernel is None:
            x = torch.arange(-radius, radius + 1, device=tensor.device, dtype=tensor.dtype)
            kernel = torch.exp(-0.5 * (x / sigma_px) ** 2)
            kernel = kernel / kernel.sum()
            self._gaussian_kernel_cache[key] = kernel
        kernel_col = kernel.view(1, 1, -1, 1)
        kernel_row = kernel.view(1, 1, 1, -1)
        channels = tensor.shape[1]
        out = F.conv2d(
            tensor,
            kernel_col.expand(channels, 1, -1, 1),
            padding=(radius, 0),
            groups=channels,
        )
        out = F.conv2d(
            out,
            kernel_row.expand(channels, 1, 1, -1),
            padding=(0, radius),
            groups=channels,
        )
        return out
