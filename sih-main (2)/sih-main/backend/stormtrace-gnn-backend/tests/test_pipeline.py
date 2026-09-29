"""Lightweight smoke tests that run fast by shrinking config before import.
Full pipeline correctness is exercised by the individual stage tests; this
file just checks the stages compose without shape errors."""
import numpy as np
from config import settings

# shrink for test speed -- must happen before importing modules that read settings at import time
settings.grid_size = 12
settings.mesh_subdivisions = 1
settings.n_ensemble_members = 3
settings.n_baseline_years = 5
settings.n_forecast_steps = 3
settings.gnn_epochs = 5
settings.unet_epochs = 5
settings.diffusion_epochs = 5
settings.diffusion_steps = 5

from src.data_gen import generate_synthetic_ensemble, generate_baseline_climatology
from src.mesh import build_icosahedral_mesh, build_sparse_adjacency, grid_to_mesh_multivar
from src.efi import compute_efi_multivar
from src.gnn import build_training_tensors, train_gnn


def test_data_gen_shapes():
    ensemble = generate_synthetic_ensemble()
    baseline = generate_baseline_climatology()
    assert ensemble.shape == (3, 3, 4, 12, 12)
    assert baseline.shape == (5, 4, 12, 12)


def test_efi_and_gnn_train_smoke():
    ensemble = generate_synthetic_ensemble(seed=1)
    baseline = generate_baseline_climatology(seed=2)
    vertices, faces = build_icosahedral_mesh()
    adjacency = build_sparse_adjacency(faces, len(vertices))

    mesh_feats = grid_to_mesh_multivar(ensemble[0].mean(axis=0), vertices)
    efi = compute_efi_multivar(mesh_feats, baseline, vertices)
    assert efi["combined_max"].shape[0] == len(vertices)

    features, labels, adjacency_sparse = build_training_tensors(mesh_feats, efi["combined_max"], adjacency)
    model, history = train_gnn(features, labels, adjacency_sparse, epochs=5)
    assert len(history) > 0
