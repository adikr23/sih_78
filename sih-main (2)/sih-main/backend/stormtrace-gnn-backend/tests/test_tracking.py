import numpy as np
from scipy.sparse import csr_matrix

from src.tracking import link_clusters_across_time, track_anomaly_4d


def test_track_creation_and_continuation():
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    frames = [
        [np.array([0, 1])],
        [np.array([0, 1])],
    ]
    tracks = link_clusters_across_time(frames, vertices)
    assert len(tracks) == 1
    assert [step for step, _ in tracks[0]["steps"]] == [0, 1]


def test_missing_frame_terminates_or_continues_by_configured_gap():
    vertices = np.array([[1.0, 0.0, 0.0], [0.99, 0.1, 0.0]])
    frames = [[np.array([0])], [], [np.array([0])]]

    assert len(link_clusters_across_time(frames, vertices, max_gap_steps=0)) == 2
    continued = link_clusters_across_time(frames, vertices, max_gap_steps=1)
    assert len(continued) == 1
    assert [step for step, _ in continued[0]["steps"]] == [0, 2]


def test_multiple_simultaneous_tracks_remain_distinct():
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.2, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.98, -0.2, 0.0],
        ]
    )
    frames = [
        [np.array([0]), np.array([2])],
        [np.array([1]), np.array([3])],
    ]
    tracks = link_clusters_across_time(frames, vertices, max_link_distance=0.5)
    assert len(tracks) == 2
    assert all(len(track["steps"]) == 2 for track in tracks)


def test_tigge_continuity_gate_rejects_distant_centroid_association():
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [np.cos(0.04), np.sin(0.04), 0.0],
        ]
    )
    frames = [[np.array([0])], [np.array([1])]]

    tracks = link_clusters_across_time(
        frames,
        vertices,
        max_link_distance=0.05,
    )

    assert len(tracks) == 1
    assert [step for step, _ in tracks[0]["steps"]] == [0, 1]

    distant_vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [np.cos(0.08), np.sin(0.08), 0.0],
        ]
    )
    distant_tracks = link_clusters_across_time(
        frames,
        distant_vertices,
        max_link_distance=0.05,
    )
    assert len(distant_tracks) == 2


def test_track_output_includes_geographic_centroid_and_points():
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],
        ]
    )
    adjacency = csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.int8))
    predictions = np.array([[0.9, 0.8], [0.8, 0.9]])
    efi = np.array([[0.7, 0.6], [0.8, 0.7]])
    dominant, tracks = track_anomaly_4d(
        predictions,
        adjacency,
        vertices,
        threshold=0.5,
        forecast_time="2025-01-01T00:00:00",
        valid_times=["2025-01-01T06:00:00", "2025-01-01T12:00:00"],
        lead_times_hours=[6, 12],
        anomaly_per_step=efi,
    )

    assert len(tracks) == 1
    assert dominant["track_id"] == "track-0001"
    assert -90 <= dominant["latitude"] <= 90
    assert -180 <= dominant["longitude"] <= 180
    assert dominant["forecast_time"] == "2025-01-01T00:00:00"
    assert dominant["valid_time"] == "2025-01-01T12:00:00"
    assert dominant["lead_time_hours"] == 12
    assert len(dominant["track_points"]) == 2
    assert dominant["confidence"] is not None
    assert dominant["track_points"][0]["anomaly_score"] is not None
