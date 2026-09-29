from config import settings

settings.grid_size = 12
settings.n_ensemble_members = 3
settings.n_baseline_years = 5
settings.n_forecast_steps = 3
settings.mesh_subdivisions = 1
settings.gnn_epochs = 5
settings.unet_epochs = 5
settings.diffusion_epochs = 5
settings.diffusion_steps = 5
settings.n_variables = 4

from src.pipeline import run_full_pipeline

result = run_full_pipeline(event_name='smoke_run', seed=7)
print(sorted(result.keys()))
print({k: result[k] for k in ['seed', 'severity', 'confidence', 'physics_score', 'efi_peak', 'unet_peak', 'diffusion_peak', 'amplitude_gain_pct', 'n_tracks', 'elapsed_seconds']})
