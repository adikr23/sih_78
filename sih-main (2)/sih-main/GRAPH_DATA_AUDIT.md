# STORMTRACE-GNN graph data audit

This inventory covers the scientific charts currently rendered by the
frontend. Values are not generated for visual smoothness; each chart either
comes from an API/cache field or from an explicit calculation documented
below. “Experimental” and “project-specific” are not verification claims.

| Graph | Source/API | Variable and unit | Aggregation/formula | Time/location | Status |
|---|---|---|---|---|---|
| GNN loss history | `PipelineResult.gnn_loss_history` from `/pipeline/run*` | training loss, dimensionless | Backend training history by epoch | Epoch order from payload | Derived training diagnostic; PASS |
| TIGGE spatio-temporal anomaly evolution | `/api/tigge/amphan/overview` | `anomaly_max`, `gnn_probability_max`, dimensionless | Backend cached per-valid-time maxima; no frontend interpolation | Actual `valid_time` and `lead_time_hours` | Experimental/derived; PASS |
| TIGGE field map | `/api/v1/forecast/field` or historical TIGGE field route | `t2m` in K, anomaly/GNN dimensionless, `tp` accumulated amount | Backend cached field; display-only spatial sampling when requested | Selected actual valid time and geographic coordinates | Real source for TIGGE fields; PASS |
| TIGGE transect | `/api/tigge/amphan/field` for selected `t2m` time plus cached IMDAA model fields | TIGGE K; U-Net/Diffusion cached model units | Nearest latitude row, linear longitude interpolation | Selected TIGGE valid time; model snapshots explicitly unmatched | Mixed real/experimental; PASS with limitation |
| Derived track table | `/api/tigge/amphan/tracks` and overview timeline | latitude/longitude degrees, anomaly score, GNN confidence | Backend centroid/association output | Actual track valid times only | Derived/experimental; PASS |
| Severity distribution | `/metrics` | alert counts | Backend `by_severity` counts | Persisted alert records | Project-specific record aggregate; PASS |
| Mesh field | `PipelineResult` plus deterministic SVG geometry | No scientific geographic quantity | Illustrative mesh topology; no lat/lon | Current pipeline result only | Illustrative; not a forecast graph |

## Unit and numerical rules

- TIGGE `t2m` remains Kelvin in API fields. Any Celsius display must use
  `K - 273.15` exactly once.
- TIGGE `u10` and `v10` are wind components in m/s. A wind-speed chart would
  require `sqrt(u10^2 + v10^2)`; no current chart labels either component as
  wind speed.
- TIGGE `tp` is accumulated precipitation. It is not displayed as mm/hour.
- Forecast x-axes use backend `valid_time`/`lead_time_hours`, not synthetic
  index-derived timestamps.
- GNN confidence is displayed as a percentage only where the backend field is
  a bounded score/probability; it remains labelled uncalibrated.
- Severity is categorical and is not converted into a numerical intensity.
- Map fields preserve actual latitude/longitude metadata and only downsample
  for visualization.
- The browser renders the numeric raster to a canvas; geographic grid labels,
  track, alert, and bounds remain SVG overlays. This changes rendering only;
  it does not alter field values or the existing visualization sampling.

## Limitations

- The current replay cache exposes ensemble-mean analysis fields for the
  visual field endpoints, not per-member time-series arrays. Therefore no
  ensemble spread chart is inferred from the cached replay.
- U-Net and diffusion charts use cached IMDAA-derived snapshots and are not
  matched TIGGE-conditioned forecasts.
- Independent high-resolution verification and forecast skill are unavailable.
