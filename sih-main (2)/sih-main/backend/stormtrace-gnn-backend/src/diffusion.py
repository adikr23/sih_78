"""
Conditional Denoising Diffusion Probabilistic Model (DDPM) prototype. In the
absence of a configured high-resolution reference, the training target is an
upsampled ensemble-member proxy, not observed 5 km data.

Upgrades over a minimal single-shot conv net:
  - a proper DDPM formulation: fixed linear beta schedule, closed-form
    q_sample (forward noising at any timestep in one shot, not a step loop),
    sinusoidal timestep embeddings (not a constant channel), and ancestral
    (not just noise-subtraction) sampling with the correct posterior variance
  - the denoiser is conditioned on the coarse-resolution field (upsampled)
    as an extra channel, so it is a genuine super-resolution/downscaling
    model, not an unconditional generator
  - physics-informed loss: alongside the standard noise-prediction MSE, a
    moisture-convergence penalty (see physics.py) is added with weight
    settings.physics_loss_weight, so the model is trained -- not just
    evaluated -- against the conservation-law check
"""
import math
import torch
import torch.nn as nn
import numpy as np

from config import settings
from src.physics import moisture_convergence_penalty


def sinusoidal_embedding(t: torch.Tensor, dim: int = 16):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConditionalDenoiser(nn.Module):
    """Input channels: [noisy_target, coarse_condition] + broadcast timestep embedding."""

    def __init__(self, t_emb_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_emb_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

        self.in_conv = nn.Conv2d(2, hidden, 3, padding=1)
        self.block1 = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU())
        self.block2 = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU())
        self.out_conv = nn.Conv2d(hidden, 1, 3, padding=1)

    def forward(self, noisy, condition, t):
        x = torch.cat([noisy, condition], dim=1)
        h = self.in_conv(x)
        t_emb = self.t_mlp(sinusoidal_embedding(t, self.t_emb_dim))  # (B, hidden)
        h = h + t_emb[:, :, None, None]
        h = self.block1(h)
        h = self.block2(h)
        return self.out_conv(h)


class DDPMSchedule:
    def __init__(self, n_steps: int, beta_start=1e-4, beta_end=2e-2):
        self.n_steps = n_steps
        self.betas = torch.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        """Closed-form forward diffusion to step t (vectorized, no step loop)."""
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise


def validate_downscaling_resolutions(input_resolution_km, target_resolution_km):
    if input_resolution_km is None or target_resolution_km is None:
        raise ValueError("true downscaling requires both input and target resolutions")
    if (
        not math.isfinite(float(input_resolution_km))
        or not math.isfinite(float(target_resolution_km))
        or input_resolution_km <= 0
        or target_resolution_km <= 0
    ):
        raise ValueError("input and target resolutions must be finite positive distances in kilometers")
    if target_resolution_km >= input_resolution_km:
        raise ValueError("true downscaling requires a finer target resolution than the input")


def train_diffusion(
    coarse_condition: torch.Tensor,
    target_high_res: torch.Tensor,
    epochs: int = None,
    target_is_reference: bool = False,
):
    """
    coarse_condition and target_high_res: matching (B,1,H,W) tensors.
    target_is_reference marks a paired, genuine higher-resolution target.
    """
    if coarse_condition.ndim != 4 or target_high_res.ndim != 4:
        raise ValueError("diffusion inputs must have shape (batch, 1, height, width)")
    if coarse_condition.shape != target_high_res.shape or coarse_condition.shape[1] != 1:
        raise ValueError("diffusion condition and target must have matching single-channel shapes")
    if target_is_reference:
        validate_downscaling_resolutions(
            settings.input_resolution_km,
            settings.target_resolution_km,
        )
    epochs = epochs or settings.diffusion_epochs
    n_steps = settings.diffusion_steps
    schedule = DDPMSchedule(n_steps)
    model = ConditionalDenoiser()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.diffusion_lr)

    for epoch in range(epochs):
        t = torch.randint(0, n_steps, (1,))
        noise = torch.randn_like(target_high_res)
        noisy = schedule.q_sample(target_high_res, t, noise)

        optimizer.zero_grad()
        predicted_noise = model(noisy, coarse_condition, t)
        noise_loss = nn.functional.mse_loss(predicted_noise, noise)

        # physics-informed term: reconstruct an approximate x0 estimate from the
        # current noise prediction and penalize conservation-law violation in it,
        # so the training signal -- not just post-hoc evaluation -- includes physics
        with torch.no_grad():
            ab = schedule.alpha_bars[t].view(-1, 1, 1, 1)
        x0_estimate = (noisy - torch.sqrt(1 - ab) * predicted_noise) / torch.sqrt(ab).clamp(min=1e-4)
        physics_penalty = moisture_convergence_penalty(x0_estimate, coarse_condition)

        loss = noise_loss + settings.physics_loss_weight * physics_penalty
        loss.backward()
        optimizer.step()

    return model, schedule


@torch.no_grad()
def sample_diffusion(model: ConditionalDenoiser, schedule: DDPMSchedule, coarse_condition: torch.Tensor):
    """Ancestral DDPM sampling: start from pure noise, iteratively denoise using the
    correct posterior mean/variance (not a naive linear noise subtraction)."""
    shape = coarse_condition.shape
    x = torch.randn(shape)

    for step in reversed(range(schedule.n_steps)):
        t = torch.tensor([step])
        beta_t = schedule.betas[step]
        alpha_t = schedule.alphas[step]
        alpha_bar_t = schedule.alpha_bars[step]

        predicted_noise = model(x, coarse_condition, t)
        mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)

        if step > 0:
            noise = torch.randn(shape)
            variance = beta_t
            x = mean + torch.sqrt(variance) * noise
        else:
            x = mean

    return x


def train_and_generate(
    ensemble_field: np.ndarray,
    variable_index: int = 0,
    target_high_res: np.ndarray = None,
    training_epochs: int = None,
):
    """
    ensemble_field: (n_members, n_vars, grid, grid) for a single forecast timestep.
    Uses the ensemble mean of `variable_index` as the coarse condition and the
    highest-amplitude member as an explicitly experimental proxy target when no
    paired high-resolution reference is supplied.

    Fields are z-score normalized before entering the diffusion process and
    de-normalized after sampling. DDPM's noise schedule assumes roughly unit-
    scale data (q_sample blends x0 with N(0,1) noise); real meteorological
    fields sit at an offset (e.g. ~25-40 degC), so without this the reverse
    process converges toward the noise prior's own scale rather than the
    field's true amplitude -- which would silently sabotage the exact
    peak-retention comparison against the U-Net baseline that is this
    project's core claim.
    """
    if ensemble_field.ndim != 4 or variable_index < 0 or variable_index >= ensemble_field.shape[1]:
        raise ValueError("ensemble_field must have shape (members, variables, latitude, longitude)")
    if settings.high_resolution_target_path and target_high_res is None:
        raise RuntimeError(
            "HIGH_RESOLUTION_TARGET_PATH is configured, but this pipeline does not load paired targets yet; "
            "refusing to train against the ensemble-member proxy"
        )
    if settings.input_resolution_km is not None and settings.target_resolution_km is not None:
        validate_downscaling_resolutions(
            settings.input_resolution_km,
            settings.target_resolution_km,
        )

    var_field = ensemble_field[:, variable_index]  # (n_members, latitude, longitude)
    extreme_member = var_field[np.abs(var_field - var_field.mean(axis=0)).sum(axis=(1, 2)).argmax()]
    coarse = torch.tensor(var_field.mean(axis=0), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    from src.unet import upscale_bilinear
    coarse_upsampled = upscale_bilinear(coarse)
    if target_high_res is None:
        target = torch.tensor(extreme_member, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        target_upsampled = upscale_bilinear(target)
        target_is_reference = False
    else:
        target_high_res = np.asarray(target_high_res)
        if target_high_res.ndim != 2 or not np.isfinite(target_high_res).all():
            raise ValueError("target_high_res must be a finite two-dimensional reference field")
        target_upsampled = torch.tensor(target_high_res, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        if coarse_upsampled.shape != target_upsampled.shape:
            raise ValueError("high-resolution target grid must align with the configured upscaled input grid")
        target_is_reference = True

    mean = target_upsampled.mean()
    std = target_upsampled.std().clamp(min=1e-3)
    coarse_norm = (coarse_upsampled - mean) / std
    target_norm = (target_upsampled - mean) / std

    model, schedule = train_diffusion(
        coarse_norm,
        target_norm,
        epochs=training_epochs,
        target_is_reference=target_is_reference,
    )
    generated_norm = sample_diffusion(model, schedule, coarse_norm)
    generated = generated_norm * std + mean

    return generated, coarse_upsampled, target_upsampled
