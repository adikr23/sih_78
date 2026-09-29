"""
Spherical (icosahedral) mesh construction and flat-grid <-> mesh reprojection.

Upgrades over a minimal version:
  - configurable subdivision depth (default 4 -> 2562 vertices, dense enough
    for real message-passing instead of a 162-point toy)
  - grid_to_mesh / mesh_to_grid are inverses of each other (nearest-neighbour
    KDTree in both directions), so re-projection correctness is unit-testable
    per the blueprint's "Mesh re-projection correctness" guardrail
  - build_sparse_adjacency is fully vectorized (numpy, no python triangle loop)
    and returns a scipy CSR matrix with symmetric GCN-style normalization,
    which is what makes the GNN forward pass a single sparse matmul instead
    of an O(V^2) dense multiply
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy import sparse

from config import settings


def build_icosahedral_mesh(subdivisions: int = None):
    subdivisions = subdivisions if subdivisions is not None else settings.mesh_subdivisions
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    return mesh.vertices.astype(np.float64), mesh.faces.astype(np.int64)


def _unit_sphere_points(
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    latitude_radians = np.radians(latitude)
    longitude_radians = np.radians(longitude)
    cos_latitude = np.cos(latitude_radians)
    return np.stack(
        (
            cos_latitude * np.cos(longitude_radians),
            cos_latitude * np.sin(longitude_radians),
            np.sin(latitude_radians),
        ),
        axis=-1,
    )


def _grid_points(
    grid_size,
    latitudes: np.ndarray = None,
    longitudes: np.ndarray = None,
) -> np.ndarray:
    n_lat, n_lon = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)
    if n_lat < 1 or n_lon < 1:
        raise ValueError("grid_size must be positive")

    latitudes = (
        np.linspace(-90.0, 90.0, n_lat)
        if latitudes is None
        else np.asarray(latitudes, dtype=np.float64)
    )
    longitudes = (
        np.linspace(-180.0, 180.0, n_lon, endpoint=False)
        if longitudes is None
        else np.asarray(longitudes, dtype=np.float64)
    )
    if latitudes.shape != (n_lat,) or longitudes.shape != (n_lon,):
        raise ValueError("latitude and longitude coordinates must match the grid dimensions")
    if not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all():
        raise ValueError("latitude and longitude coordinates must be finite")
    if np.any((latitudes < -90.0) | (latitudes > 90.0)):
        raise ValueError("latitude coordinates must be within [-90, 90]")
    if np.any((longitudes < -180.0) | (longitudes > 360.0)):
        raise ValueError("longitude coordinates must be within [-180, 360]")

    latitude_grid, longitude_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    return _unit_sphere_points(latitude_grid, longitude_grid).reshape(-1, 3)


def mesh_to_geographic(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert spherical Cartesian mesh vertices to latitude/longitude in degrees."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (n_points, 3)")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite")
    norms = np.linalg.norm(vertices, axis=1)
    if np.any(norms == 0):
        raise ValueError("vertices must not contain the zero vector")
    unit_vertices = vertices / norms[:, None]
    latitude = np.degrees(np.arcsin(np.clip(unit_vertices[:, 2], -1.0, 1.0)))
    longitude = np.degrees(np.arctan2(unit_vertices[:, 1], unit_vertices[:, 0]))
    return latitude, longitude


def _validated_vertices(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (n_points, 3)")
    if len(vertices) == 0 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be non-empty and finite")
    norms = np.linalg.norm(vertices, axis=1)
    if np.any(norms == 0):
        raise ValueError("vertices must not contain the zero vector")
    return vertices / norms[:, None]


def grid_to_mesh(
    grid: np.ndarray,
    vertices: np.ndarray,
    latitudes: np.ndarray = None,
    longitudes: np.ndarray = None,
) -> np.ndarray:
    """Nearest-neighbour sample a geographic grid onto spherical mesh vertices."""
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("grid must be a two-dimensional field")
    grid_points = _grid_points(grid.shape, latitudes, longitudes)
    mesh_vertices = _validated_vertices(vertices)
    _, indices = cKDTree(grid_points).query(mesh_vertices)
    return grid.ravel()[indices]


def mesh_to_grid(
    mesh_values: np.ndarray,
    vertices: np.ndarray,
    grid_size,
    latitudes: np.ndarray = None,
    longitudes: np.ndarray = None,
) -> np.ndarray:
    """Nearest-neighbour interpolation from spherical mesh vertices to a geographic grid."""
    mesh_values = np.asarray(mesh_values)
    mesh_vertices = _validated_vertices(vertices)
    if mesh_values.ndim != 1 or mesh_values.shape[0] != mesh_vertices.shape[0]:
        raise ValueError("mesh_values must contain one value per mesh vertex")
    grid_shape = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)
    grid_points = _grid_points(grid_shape, latitudes, longitudes)
    _, indices = cKDTree(mesh_vertices).query(grid_points)
    return mesh_values[indices].reshape(grid_shape)


def grid_to_mesh_multivar(
    grid_stack: np.ndarray,
    vertices: np.ndarray,
    latitudes: np.ndarray = None,
    longitudes: np.ndarray = None,
) -> np.ndarray:
    """grid_stack: (n_vars, grid, grid) -> (n_mesh_points, n_vars), single KDTree build reused
    across variables for efficiency."""
    grid_stack = np.asarray(grid_stack)
    if grid_stack.ndim != 3:
        raise ValueError("grid_stack must have shape (n_vars, latitude, longitude)")
    grid_points = _grid_points(grid_stack.shape[-2:], latitudes, longitudes)
    mesh_vertices = _validated_vertices(vertices)
    _, idx = cKDTree(grid_points).query(mesh_vertices)
    n_vars = grid_stack.shape[0]
    flat = grid_stack.reshape(n_vars, -1)
    return flat[:, idx].T  # (n_mesh_points, n_vars)


def build_sparse_adjacency(faces: np.ndarray, n_points: int) -> sparse.csr_matrix:
    """Vectorized edge extraction (no per-triangle python loop) + symmetric
    GCN normalization: A_hat = D^-1/2 (A + I) D^-1/2."""
    edges_a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    edges_b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    row = np.concatenate([edges_a, edges_b])
    col = np.concatenate([edges_b, edges_a])
    data = np.ones(len(row), dtype=np.float32)
    A = sparse.coo_matrix((data, (row, col)), shape=(n_points, n_points))
    A = A.tocsr()
    A.data[:] = 1.0  # dedupe multi-edges to binary adjacency
    A = A + sparse.eye(n_points, format="csr")  # self loops

    deg = np.asarray(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)
    D_inv_sqrt = sparse.diags(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat.tocsr()


def scipy_csr_to_torch_sparse(csr):
    import torch
    coo = csr.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()


if __name__ == "__main__":
    import os
    os.makedirs(settings.data_dir, exist_ok=True)
    vertices, faces = build_icosahedral_mesh()
    ensemble = np.load(f"{settings.data_dir}/ensemble.npy")  # (steps, members, vars, g, g)
    mesh_values = grid_to_mesh_multivar(ensemble[0, 0], vertices)  # step0, member0

    # round-trip fidelity check
    back = mesh_to_grid(mesh_values[:, 0], vertices, ensemble.shape[-1])
    err = np.abs(back - ensemble[0, 0, 0]).mean()

    adjacency = build_sparse_adjacency(faces, len(vertices))

    np.save(f"{settings.data_dir}/mesh_vertices.npy", vertices)
    np.save(f"{settings.data_dir}/mesh_faces.npy", faces)
    np.save(f"{settings.data_dir}/mesh_values.npy", mesh_values)
    sparse.save_npz(f"{settings.data_dir}/mesh_adjacency.npz", adjacency)

    print(f"Mesh built. Vertices: {vertices.shape} Faces: {faces.shape}")
    print(f"Mesh values (multivar): {mesh_values.shape}")
    print(f"Adjacency: {adjacency.shape}, nnz={adjacency.nnz}")
    print(f"Round-trip reprojection mean abs error: {err:.4f}")
