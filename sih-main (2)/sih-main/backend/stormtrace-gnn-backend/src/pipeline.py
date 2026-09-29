"""
Wires every stage into one run, mirroring blueprint Figure 2/3:
  ingest -> mesh projection -> EFI -> GNN (per timestep) -> 4D tracking ->
  crop -> U-Net baseline + diffusion downscaling -> physics check -> severity
  -> persist -> return.

Efficiency choices (this is the "make it efficiently heavy" part):
  - the icosahedral mesh + sparse adjacency are expensive to build and are
    IDENTICAL across every pipeline run (the mesh doesn't depend on the
    forecast data), so they are built once per process and cached at module
    scope (`_MESH_CACHE`) instead of rebuilt on every request
  - GNN inference across the 8 forecast timesteps reuses one trained model
    (trained once on the timestep with the strongest EFI signal, per the
    blueprint's guardrail "never require end-to-end joint training for the
    MVP" -- Stage 1 and Stage 2 stay separate models with a fixed interface)
  - all per-timestep GNN forward passes are pure sparse matmuls (see gnn.py),
    so scoring all 8 timesteps costs roughly the same as scoring one dense
    step would with a naive implementation
"""
import time
import numpy as np
import torch
from scipy import sparse

from config import settings
from src.data_provider import canonicalize_weather_data, get_data_provider
from src.mesh import (
    build_icosahedral_mesh,
    build_sparse_adjacency,
    grid_to_mesh_multivar,
    scipy_csr_to_torch_sparse,
)
from src.efi import compute_efi_multivar
from src.gnn import AnomalyGNN, build_training_tensors, train_gnn
from src.tracking import track_anomaly_4d
from src.unet import train_and_run_unet
from src.diffusion import train_and_generate
from src.physics import physics_consistency_score, moisture_convergence_score_numpy
from src.severity import compute_alert
from src.db import init_db, save_alert

_MESH_CACHE = {}


def _get_mesh():
    if not _MESH_CACHE:
        vertices, faces = build_icosahedral_mesh()
        adjacency_csr = build_sparse_adjacency(faces, len(vertices))
        adjacency_sparse_torch = scipy_csr_to_torch_sparse(adjacency_csr)
        _MESH_CACHE.update(
            vertices=vertices, faces=faces,
            adjacency_csr=adjacency_csr, adjacency_sparse_torch=adjacency_sparse_torch,
        )
    return _MESH_CACHE


def run_full_pipeline(event_name: str = "live_run", seed: int = None, progress_cb=None) -> dict:
    def report(msg):
        if progress_cb:
            progress_cb(msg)

    t0 = time.time()
    seed = seed if seed is not None else int(time.time()) % 100000
    provider = get_data_provider()
    provider_meta = provider.metadata

    report("1/8 generating ensemble + baseline")
    ensemble = provider.generate_ensemble(seed=seed)          # (steps, members, vars, g, g)
    baseline = provider.generate_baseline(seed=seed + 1)    # (years, vars, g, g)
    weather_fields = canonicalize_weather_data(ensemble, baseline, provider)
    provider_meta = weather_fields.provenance
    ensemble = weather_fields.ensemble
    baseline = weather_fields.climatology

    report("2/8 building / loading cached spherical mesh")
    mesh = _get_mesh()
    vertices, adjacency_csr = mesh["vertices"], mesh["adjacency_csr"]
    adjacency_sparse_torch = mesh["adjacency_sparse_torch"]

    report("3/8 projecting ensemble mean onto mesh + computing EFI per timestep")
    n_steps = ensemble.shape[0]
    efi_per_step = []      # combined_max EFI array per timestep, shape (n_steps, n_points)
    mesh_feats_per_step = []
    for t in range(n_steps):
        ens_mean_t = ensemble[t].mean(axis=0)  # (n_vars, g, g)
        mesh_feats_t = grid_to_mesh_multivar(
            ens_mean_t, vertices, weather_fields.latitudes, weather_fields.longitudes
        )  # (n_points, n_vars)
        efi_t = compute_efi_multivar(
            mesh_feats_t,
            baseline,
            vertices,
            weather_fields.latitudes,
            weather_fields.longitudes,
        )
        efi_per_step.append(efi_t["combined_max"])
        mesh_feats_per_step.append(mesh_feats_t)
    efi_per_step = np.stack(efi_per_step)  # (n_steps, n_points)

    report("4/8 training GNN on the peak-signal timestep, scoring all timesteps")
    peak_t = int(efi_per_step.max(axis=1).argmax())
    features, labels, _ = build_training_tensors(mesh_feats_per_step[peak_t], efi_per_step[peak_t], adjacency_csr)
    model, loss_history = train_gnn(features, labels, adjacency_sparse_torch)

    predictions_per_step = np.zeros((n_steps, len(vertices)), dtype=np.float32)
    with torch.no_grad():
        for t in range(n_steps):
            feats_t = torch.tensor(mesh_feats_per_step[t], dtype=torch.float32)
            predictions_per_step[t] = model(feats_t, adjacency_sparse_torch).numpy()

    report("5/8 linking anomaly clusters into a 4D bounding box")
    bbox, all_tracks = track_anomaly_4d(
        predictions_per_step,
        adjacency_csr,
        vertices,
        threshold=settings.efi_anomaly_threshold,
        max_gap_steps=settings.max_track_gap_steps,
        forecast_time=provider_meta.get("initialization_time"),
        valid_times=provider_meta.get("valid_time"),
        lead_times_hours=provider_meta.get("lead_time"),
        efi_per_step=efi_per_step,
    )
    if bbox is None:
        # No anomaly cleared the threshold this run -- still return a well-formed, low-severity result
        bbox = {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0, "z_min": 0, "z_max": 0,
                "t_start": 0, "t_end": n_steps - 1, "n_points": 0}

    report("6/8 downscaling: U-Net baseline vs conditional diffusion")
    crop_step = bbox["t_end"] if bbox["n_points"] > 0 else peak_t
    unet_input = torch.tensor(
        ensemble[crop_step].mean(axis=0)[0:1], dtype=torch.float32  # variable 0 = t2m
    ).unsqueeze(0)
    unet_output = train_and_run_unet(unet_input)

    diffusion_output, coarse_upsampled, _ = train_and_generate(ensemble[crop_step], variable_index=0)
    downscaling_target = {
        "status": "prototype_proxy",
        "target_type": "selected_ensemble_member_proxy",
        "validated": False,
        "input_resolution_km": (
            settings.input_resolution_km or settings.data_spatial_resolution_km
        ),
        "configured_target_resolution_km": settings.target_resolution_km,
        "actual_target_resolution_km": None,
        "high_resolution_reference_configured": bool(settings.high_resolution_target_path),
    }

    report("7/8 physics consistency check")
    physics_score_agreement = physics_consistency_score(
        diffusion_output.detach().numpy(), ensemble[crop_step, :, 0]
    )
    physics_score_conservation = moisture_convergence_score_numpy(
        diffusion_output.detach().numpy(), coarse_upsampled.detach().numpy()
    )
    physics_score = round(0.5 * physics_score_agreement + 0.5 * physics_score_conservation, 3)
    from src.physics import physics_constraint_diagnostic
    physics_diagnostic = physics_constraint_diagnostic(
        variable_names=weather_fields.variables,
        variable_units=weather_fields.units,
        has_geo_coordinates=True,
    )

    report("8/8 computing severity-tiered alert and persisting")
    efi_peak = float(efi_per_step[crop_step].max())
    alert = compute_alert(diffusion_output.detach().numpy(), efi_peak, physics_score)

    unet_peak = float(unet_output.max().item())
    diffusion_peak = float(diffusion_output.max().item())
    amplitude_gain_pct = (
        round((diffusion_peak - unet_peak) / (abs(unet_peak) + 1e-6) * 100, 2)
    )

    alert.update(
        physics_score=physics_score,
        efi_peak=round(efi_peak, 3),
        unet_peak=round(unet_peak, 3),
        diffusion_peak=round(diffusion_peak, 3),
        amplitude_gain_pct=amplitude_gain_pct,
        bounding_box=bbox,
        n_tracks=len(all_tracks),
        gnn_loss_history=loss_history,
        seed=seed,
        elapsed_seconds=round(time.time() - t0, 2),
        source_provider=provider_meta["provider"],
        source_label=provider_meta["label"],
        source_description=provider_meta["description"],
        data_source=provider_meta["label"],
        data_provenance=provider_meta,
        physics_constraint=physics_diagnostic,
        downscaling=downscaling_target,
    )

    init_db()
    record = save_alert(alert, event_name=event_name, is_historical_replay=(event_name != "live_run"))
    alert["alert_id"] = record.id
    return alert


if __name__ == "__main__":
    result = run_full_pipeline(event_name="cli_run", progress_cb=print)
    print("FINAL RESULT:", result)
