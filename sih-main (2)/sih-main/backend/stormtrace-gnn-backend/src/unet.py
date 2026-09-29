"""
Baseline CNN/U-Net downscaler. Per blueprint section 3.2 this exists on purpose
as the spectral-smoothing benchmark to measure the diffusion model against --
it is trained to minimize pixel-wise MSE, which is exactly why it regresses to
the mean and flattens extremes. Kept deliberately simple; the point of Stage 2
is that the diffusion model in diffusion.py beats this, not that this is good.
"""
import torch
import torch.nn as nn

from config import settings


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, in_channels, 3, padding=1),
        )

    def forward(self, x):
        d = self.down(x)
        b = self.bottleneck(d)
        return self.up(d + b)  # skip connection


def upscale_bilinear(grid_tensor: torch.Tensor, scale: int = None):
    scale = scale or settings.upscale_factor
    return nn.functional.interpolate(grid_tensor, scale_factor=scale, mode="bilinear", align_corners=False)


def train_and_run_unet(low_res_field: torch.Tensor, epochs: int = None) -> torch.Tensor:
    """low_res_field: (1,1,H,W). Returns the upscaled, denoised-by-averaging output."""
    epochs = epochs or settings.unet_epochs
    target = upscale_bilinear(low_res_field)
    upscaled_input = upscale_bilinear(low_res_field)

    model = SimpleUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(upscaled_input)
        loss = loss_fn(pred, target)  # trained to match the mean -> smooths peaks by design
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return model(upscaled_input)
