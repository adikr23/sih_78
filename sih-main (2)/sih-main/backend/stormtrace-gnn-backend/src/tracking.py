"""
Stage 1, temporal half: turn per-timestep GNN anomaly scores into a single
4D (x, y, z, t) bounding box tracking the anomaly across the D+3..D+10 window,
per blueprint section 3.1 ("4D temporal bounding box... reusing standard
multi-object tracking association logic: nearest-neighbour or Hungarian
matching on mesh-node clusters").

Pipeline:
  1. At each timestep, connected-component cluster the mesh nodes whose GNN
     score exceeds the threshold (using the sparse adjacency, via scipy's
     connected_components -- vectorized, not a manual BFS).
  2. Take the largest cluster per timestep as "the" tracked anomaly (extendable
     to multi-object tracking by keeping all clusters and doing Hungarian
     matching across ALL of them, which is implemented below and used when
     more than one significant cluster exists at a timestep).
  3. Link cluster centroids across consecutive timesteps by nearest-centroid
     (Hungarian assignment, scipy.optimize.linear_sum_assignment) to build a
     continuous trajectory even if the anomaly moves.
  4. Union the bounding volumes across all linked timesteps into one 4D box.
"""
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linear_sum_assignment
from src.mesh import mesh_to_geographic


def cluster_anomalies(predictions: np.ndarray, adjacency_csr, threshold: float):
    """Connected-component clustering restricted to the anomalous subgraph."""
    mask = predictions > threshold
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

    sub_adj = adjacency_csr[idx][:, idx]
    n_components, labels = connected_components(sub_adj, directed=False)

    clusters = []
    for c in range(n_components):
        member_idx = idx[labels == c]
        if len(member_idx) == 0:
            continue
        clusters.append(member_idx)
    return clusters


def _cluster_centroid(vertices, cluster_idx):
    centroid = vertices[cluster_idx].mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm <= 1e-12:
        raise ValueError("anomaly cluster has no stable spherical centroid")
    return centroid / norm


def link_clusters_across_time(
    per_step_clusters,
    vertices,
    max_link_distance=1.5,
    max_gap_steps=0,
):
    """
    per_step_clusters: list (len = n_steps) of lists of index arrays (clusters at that step)
    Greedy/Hungarian-matched trajectory linking: each step's clusters are matched to the
    previous step's active tracks by Euclidean chord distance between unit-sphere
    centroids; unmatched clusters start new tracks.
    Returns list of tracks, each track = list of (t, cluster_idx_array).
    """
    if max_gap_steps < 0:
        raise ValueError("max_gap_steps must be non-negative")
    active_tracks = []
    finished_tracks = []

    for t, clusters in enumerate(per_step_clusters):
        still_active = []
        for track in active_tracks:
            if t - track["last_t"] <= max_gap_steps + 1:
                still_active.append(track)
            else:
                finished_tracks.append(track)
        active_tracks = still_active

        if not clusters:
            continue
        centroids = np.array([_cluster_centroid(vertices, c) for c in clusters])

        if not active_tracks:
            for c, cen in zip(clusters, centroids):
                active_tracks.append({"last_centroid": cen, "last_t": t, "steps": [(t, c)]})
            continue

        prev_centroids = np.array([tr["last_centroid"] for tr in active_tracks])
        cost = np.linalg.norm(prev_centroids[:, None, :] - centroids[None, :, :], axis=-1)
        row_idx, col_idx = linear_sum_assignment(cost)

        matched_cols = set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] <= max_link_distance:
                active_tracks[r]["steps"].append((t, clusters[c]))
                active_tracks[r]["last_centroid"] = centroids[c]
                active_tracks[r]["last_t"] = t
                matched_cols.add(c)

        # unmatched new clusters become new tracks
        for c_i, (c, cen) in enumerate(zip(clusters, centroids)):
            if c_i not in matched_cols:
                active_tracks.append({"last_centroid": cen, "last_t": t, "steps": [(t, c)]})

    finished_tracks.extend(active_tracks)
    return finished_tracks


def _at(values, index):
    if values is None:
        return None
    if isinstance(values, (list, tuple, np.ndarray)):
        if index >= len(values):
            return None
        value = values[index]
    else:
        value = values
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="s")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _geographic_bbox(longitudes, latitudes):
    sorted_longitudes = np.sort(np.mod(longitudes, 360.0))
    if len(sorted_longitudes) == 1:
        west = east = float(sorted_longitudes[0])
    else:
        gaps = np.diff(np.append(sorted_longitudes, sorted_longitudes[0] + 360.0))
        largest_gap = int(np.argmax(gaps))
        west = float(sorted_longitudes[(largest_gap + 1) % len(sorted_longitudes)])
        east = float(sorted_longitudes[largest_gap])
    west = (west + 180.0) % 360.0 - 180.0
    east = (east + 180.0) % 360.0 - 180.0
    return {
        "latitude_min": float(np.min(latitudes)),
        "latitude_max": float(np.max(latitudes)),
        "longitude_west": west,
        "longitude_east": east,
        "crosses_dateline": west > east,
    }


def track_to_bbox4d(
    track,
    vertices,
    track_id=None,
    forecast_time=None,
    valid_times=None,
    lead_times_hours=None,
    efi_per_step=None,
    anomaly_per_step=None,
    confidence_per_step=None,
    severity_per_step=None,
):
    all_idx = np.concatenate([idx for _, idx in track["steps"]])
    unique_idx = np.unique(all_idx)
    points = vertices[unique_idx]
    t_values = [t for t, _ in track["steps"]]
    latitude, longitude = mesh_to_geographic(points)
    centroid = points.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    centroid_latitude, centroid_longitude = mesh_to_geographic(centroid.reshape(1, 3))
    track_points = []
    efi_values = []
    confidence_values = []
    for timestep, cluster_idx in track["steps"]:
        cluster_latitude, cluster_longitude = mesh_to_geographic(vertices[cluster_idx])
        cluster_centroid = _cluster_centroid(vertices, cluster_idx).reshape(1, 3)
        point_latitude, point_longitude = mesh_to_geographic(cluster_centroid)
        confidence = (
            float(np.mean(confidence_per_step[timestep][cluster_idx]))
            if confidence_per_step is not None
            else None
        )
        efi = (
            float(np.mean(efi_per_step[timestep][cluster_idx]))
            if efi_per_step is not None
            else None
        )
        anomaly_score = (
            float(np.mean(anomaly_per_step[timestep][cluster_idx]))
            if anomaly_per_step is not None
            else None
        )
        if efi is not None:
            efi_values.append(efi)
        if confidence is not None:
            confidence_values.append(confidence)
        track_points.append(
            {
                "forecast_time": _at(forecast_time, timestep),
                "valid_time": _at(valid_times, timestep),
                "lead_time_hours": _at(lead_times_hours, timestep),
                "latitude": float(point_latitude[0]),
                "longitude": float(point_longitude[0]),
                "efi": efi,
                "anomaly_score": anomaly_score,
                "confidence": confidence,
                "severity": _at(severity_per_step, timestep),
                "geographic_bbox": _geographic_bbox(cluster_longitude, cluster_latitude),
            }
        )

    latest_timestep = max(t_values)
    return {
        "x_min": float(points[:, 0].min()), "x_max": float(points[:, 0].max()),
        "y_min": float(points[:, 1].min()), "y_max": float(points[:, 1].max()),
        "z_min": float(points[:, 2].min()), "z_max": float(points[:, 2].max()),
        "t_start": int(min(t_values)),
        "t_end": int(max(t_values)),
        "n_points": int(len(unique_idx)),
        "track_id": track_id,
        "forecast_time": _at(forecast_time, latest_timestep),
        "valid_time": _at(valid_times, latest_timestep),
        "lead_time_hours": _at(lead_times_hours, latest_timestep),
        "latitude": float(centroid_latitude[0]),
        "longitude": float(centroid_longitude[0]),
        "efi": float(np.mean(efi_values)) if efi_values else None,
        "confidence": float(np.mean(confidence_values)) if confidence_values else None,
        "severity": _at(severity_per_step, latest_timestep),
        "geographic_bbox": _geographic_bbox(longitude, latitude),
        "track_points": track_points,
    }


def track_anomaly_4d(
    predictions_per_step,
    adjacency_csr,
    vertices,
    threshold: float,
    max_gap_steps: int = 0,
    forecast_time=None,
    valid_times=None,
    lead_times_hours=None,
    efi_per_step=None,
    anomaly_per_step=None,
    severity_per_step=None,
    max_link_distance=None,
):
    """
    predictions_per_step: (n_steps, n_points) GNN anomaly scores at each forecast step
    Returns the dominant 4D bounding box (largest track by total point-count) plus
    the full list of tracks for completeness.
    """
    per_step_clusters = [
        cluster_anomalies(predictions_per_step[t], adjacency_csr, threshold)
        for t in range(predictions_per_step.shape[0])
    ]
    tracks = link_clusters_across_time(
        per_step_clusters,
        vertices,
        max_link_distance=1.5 if max_link_distance is None else max_link_distance,
        max_gap_steps=max_gap_steps,
    )
    if not tracks:
        return None, []

    boxes = [
        track_to_bbox4d(
            track,
            vertices,
            track_id=f"track-{index + 1:04d}",
            forecast_time=forecast_time,
            valid_times=valid_times,
            lead_times_hours=lead_times_hours,
            efi_per_step=efi_per_step,
            anomaly_per_step=anomaly_per_step,
            confidence_per_step=predictions_per_step,
            severity_per_step=severity_per_step,
        )
        for index, track in enumerate(tracks)
    ]
    dominant_idx = int(np.argmax([b["n_points"] for b in boxes]))
    return boxes[dominant_idx], boxes
