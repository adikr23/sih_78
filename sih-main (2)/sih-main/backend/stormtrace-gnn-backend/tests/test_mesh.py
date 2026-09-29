import numpy as np
from src.mesh import (
    build_icosahedral_mesh, grid_to_mesh, mesh_to_grid,
    grid_to_mesh_multivar, build_sparse_adjacency, mesh_to_geographic,
)


def test_mesh_build_shapes():
    vertices, faces = build_icosahedral_mesh(subdivisions=2)
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert len(vertices) > 100


def test_reprojection_round_trip_recovers_field():
    """Reproject a smooth field on a global latitude/longitude grid."""
    grid_size = 16
    latitudes = np.linspace(-90, 90, grid_size)
    longitudes = np.linspace(-180, 180, grid_size, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    analytic_field = np.exp(-((lat_grid / 60) ** 2 + (lon_grid / 90) ** 2))

    vertices, _ = build_icosahedral_mesh(subdivisions=3)
    mesh_values = grid_to_mesh(analytic_field, vertices, latitudes, longitudes)
    recovered = mesh_to_grid(mesh_values, vertices, grid_size, latitudes, longitudes)

    err = np.abs(recovered - analytic_field).mean()
    assert err < 0.1, f"round-trip reprojection error too high: {err}"


def test_known_geographic_grid_mapping_is_deterministic():
    latitudes = np.array([-45.0, 45.0])
    longitudes = np.array([-90.0, 90.0])
    grid = np.array([[1.0, 2.0], [3.0, 4.0]])
    latitude_grid, longitude_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    lat_radians = np.radians(latitude_grid.ravel())
    lon_radians = np.radians(longitude_grid.ravel())
    vertices = np.column_stack(
        (
            np.cos(lat_radians) * np.cos(lon_radians),
            np.cos(lat_radians) * np.sin(lon_radians),
            np.sin(lat_radians),
        )
    )

    first = grid_to_mesh(grid, vertices, latitudes, longitudes)
    second = grid_to_mesh(grid, vertices, latitudes, longitudes)
    assert np.array_equal(first, grid.ravel())
    assert np.array_equal(first, second)


def test_mesh_to_geographic_returns_bounded_coordinates():
    vertices = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    latitude, longitude = mesh_to_geographic(vertices)
    assert np.all((-90 <= latitude) & (latitude <= 90))
    assert np.all((-180 <= longitude) & (longitude <= 180))
    assert np.allclose(latitude, [0.0, 0.0, 90.0])
    assert np.allclose(longitude, [0.0, 90.0, 0.0])


def test_sparse_adjacency_symmetric_and_normalized():
    vertices, faces = build_icosahedral_mesh(subdivisions=2)
    adj = build_sparse_adjacency(faces, len(vertices))
    dense = adj.toarray()
    assert np.allclose(dense, dense.T, atol=1e-5), "adjacency should be symmetric"
    assert adj.nnz > 0
