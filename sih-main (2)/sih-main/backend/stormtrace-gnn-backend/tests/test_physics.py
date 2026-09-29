import numpy as np
import pytest
import torch
from src.physics import (
    moisture_convergence_penalty,
    moisture_flux_convergence,
    physics_consistency_score,
    physics_constraint_diagnostic,
)


def test_unsupported_peak_penalized_more_than_supported_peak():
    """Guardrail from blueprint section 13: a synthetic field with rainfall but no
    moisture convergence is penalized more than a physically consistent field."""
    # a coarse condition field with a real local convergence pocket (negative Laplacian)
    # centered at the same location as the generated peak
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    convergent_condition = torch.exp(-(((xx - 8) ** 2 + (yy - 8) ** 2) / 6.0)).unsqueeze(0).unsqueeze(0)
    flat_condition = torch.zeros(1, 1, 16, 16)  # no convergence anywhere

    spike = torch.zeros(1, 1, 16, 16)
    spike[0, 0, 8, 8] = 10.0  # sharp peak at the same location for both cases

    penalty_unsupported = moisture_convergence_penalty(spike, flat_condition).item()
    penalty_supported = moisture_convergence_penalty(spike, convergent_condition).item()

    assert penalty_unsupported > penalty_supported


def test_physics_consistency_score_bounded():
    generated = np.random.rand(16, 16)
    ensemble = np.random.rand(5, 8, 8)
    score = physics_consistency_score(generated, ensemble)
    assert 0.0 <= score <= 1.0


def test_mfc_has_expected_shape_and_is_deterministic():
    latitude = np.linspace(-30, 30, 5)
    longitude = np.linspace(0, 40, 6)
    humidity = torch.ones(1, 1, 5, 6) * 0.01
    wind_u = torch.ones_like(humidity) * 5
    wind_v = torch.ones_like(humidity) * 2
    first = moisture_flux_convergence(humidity, wind_u, wind_v, latitude, longitude)
    second = moisture_flux_convergence(humidity, wind_u, wind_v, latitude, longitude)

    assert first.shape == humidity.shape
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_mfc_rejects_missing_physical_inputs_and_bad_shapes():
    field = torch.ones(1, 1, 4, 4)
    with pytest.raises(ValueError, match="together"):
        moisture_convergence_penalty(field, field, specific_humidity=field)
    with pytest.raises(ValueError, match="must share shape"):
        moisture_flux_convergence(field, field[:, :, :3], field, np.arange(4), np.arange(4))


def test_physics_diagnostic_labels_laplacian_fallback():
    diagnostic = physics_constraint_diagnostic(
        variable_names=("t2m", "wind_u", "wind_v"),
        variable_units=(None, None, None),
        has_geo_coordinates=True,
    )
    assert diagnostic["method"] == "laplacian_proxy"
    assert diagnostic["available"] is False
    assert "specific_humidity (q)" in diagnostic["missing_variables"]
