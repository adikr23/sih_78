"""
Synthetic replacement for NEPS-G (12km global ensemble) + ERA5/IMDAA baseline.

Upgrades over a toy single-variable/single-timestep generator:
  - 4 physical variables per cell: t2m, rainfall proxy, wind_u, wind_v
  - full temporal window D+3..D+10 (n_forecast_steps), with the anomaly
    advecting (moving) across the grid over time so Stage-1 tracking has
    something real to link across steps
  - vectorized (no python loops over members/years)

Shapes:
  ensemble:  (n_steps, n_members, n_vars, grid, grid)
  baseline:  (n_years, n_vars, grid, grid)
"""
import numpy as np
from config import settings


def _hotspot_field(grid_size: int, cx: float, cy: float, amp: float, spread: float):
    x, y = np.meshgrid(
        np.linspace(-3, 3, grid_size), np.linspace(-3, 3, grid_size)
    )
    return amp * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / spread))


def generate_synthetic_ensemble(
    n_steps: int = None,
    n_members: int = None,
    grid_size: int = None,
    seed: int = 0,
):
    n_steps = n_steps or settings.n_forecast_steps
    n_members = n_members or settings.n_ensemble_members
    grid_size = grid_size or settings.grid_size
    rng = np.random.default_rng(seed)

    # anomaly advects diagonally across the domain over the forecast window
    track_x = np.linspace(-2.0, 2.0, n_steps)
    track_y = np.linspace(-1.5, 1.5, n_steps)

    out = np.zeros((n_steps, n_members, settings.n_variables, grid_size, grid_size), dtype=np.float32)

    for t in range(n_steps):
        # variable 0: temperature, deg C, base 25 with a moving hot dome
        t2m_base = 25 + 15 * _hotspot_field(grid_size, track_x[t], track_y[t], 1.0, 3.0)
        # variable 1: rainfall proxy, mm/hr, correlated with a moisture pocket slightly offset
        rain_base = 40 * _hotspot_field(grid_size, track_x[t] + 0.3, track_y[t] - 0.2, 1.0, 1.8)
        rain_base = np.clip(rain_base, 0, None)
        # variable 2/3: wind u/v (m/s), converging toward the anomaly center (feeds physics check)
        x, y = np.meshgrid(np.linspace(-3, 3, grid_size), np.linspace(-3, 3, grid_size))
        dx, dy = (track_x[t] - x), (track_y[t] - y)
        dist = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        wind_u = 6 * dx / dist * np.exp(-dist / 2.5)
        wind_v = 6 * dy / dist * np.exp(-dist / 2.5)

        for m in range(n_members):
            member_noise_scale = 1.0 + 0.15 * m  # spread grows across members, like a real EPS
            out[t, m, 0] = t2m_base + rng.normal(0, 1.5 * member_noise_scale, (grid_size, grid_size))
            out[t, m, 1] = np.clip(
                rain_base + rng.normal(0, 3.0 * member_noise_scale, (grid_size, grid_size)), 0, None
            )
            out[t, m, 2] = wind_u + rng.normal(0, 0.8 * member_noise_scale, (grid_size, grid_size))
            out[t, m, 3] = wind_v + rng.normal(0, 0.8 * member_noise_scale, (grid_size, grid_size))

    return out


def generate_baseline_climatology(grid_size: int = None, n_years: int = None, seed: int = 1):
    grid_size = grid_size or settings.grid_size
    n_years = n_years or settings.n_baseline_years
    rng = np.random.default_rng(seed)

    baseline = np.zeros((n_years, settings.n_variables, grid_size, grid_size), dtype=np.float32)
    baseline[:, 0] = 25 + rng.normal(0, 2.0, (n_years, grid_size, grid_size))       # t2m
    baseline[:, 1] = np.clip(2 + rng.normal(0, 2.0, (n_years, grid_size, grid_size)), 0, None)  # rainfall
    baseline[:, 2] = rng.normal(0, 1.2, (n_years, grid_size, grid_size))            # wind_u
    baseline[:, 3] = rng.normal(0, 1.2, (n_years, grid_size, grid_size))            # wind_v
    return baseline


if __name__ == "__main__":
    import os
    os.makedirs(settings.data_dir, exist_ok=True)
    ensemble = generate_synthetic_ensemble()
    baseline = generate_baseline_climatology()
    np.save(f"{settings.data_dir}/ensemble.npy", ensemble)
    np.save(f"{settings.data_dir}/baseline.npy", baseline)
    print("Saved ensemble.npy", ensemble.shape, "and baseline.npy", baseline.shape)
