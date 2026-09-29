# Track Geographic Audit

## Verdict: PARTIALLY VERIFIED

The cached track points use geographic coordinates from the TIGGE analysis grid,
and their valid times agree with their forecast lead times. The old association
gate nevertheless linked unrelated detections across implausibly large
distances. It has been tightened and the local replay cache rebuilt. The result
is a more spatially constrained **experimental GNN anomaly-cluster track**, not
a verified cyclone track. GNN calibration, component shape, and independent
track truth remain unavailable.

## Data and coordinate checks

- Source: local NCMRWF TIGGE Amphan forecast initialized at 2020-05-16 00 UTC.
- The input GRIB has 41 lead times and 11 perturbed members. The replay retains
  these forecast times; its common TIGGE/IMDAA analysis crop is 209 × 139,
  latitude 30.00°N down to 5.04°N and longitude 75.06°E to 99.90°E. The crop
  differs from the full 209 × 195 source grid.
- The cached time axis matches the replay timeline. All 328 track detections
  map to a cached field timestep, and each point satisfies
  `valid_time = initialization_time + lead_time_hours`. No coordinates are
  outside latitude/longitude ranges.
- TIGGE latitude is descending and longitude is ascending. The raster preserves
  the source row order (north at the top); the SVG overlays project longitude
  to x and invert latitude to y. Track points are serialized as separate
  latitude/longitude fields, converted from unit-sphere centroids, and retained
  in chronological order. No pixel or grid indices are used as geographic
  coordinates.

## Association correction and measured result

The previous cache used a 0.25 Euclidean chord-distance gate between normalized
unit-sphere centroids. That corresponds to approximately 1,594 km of
great-circle distance and allowed unrelated clusters to become one displayed
path. In that cache, 72 of 241 within-track links exceeded 500 km, 23 exceeded
1,000 km, and the maximum was 1,574.68 km in six hours.

The TIGGE replay now uses a 0.05 chord-distance gate (approximately 318.6 km
great-circle distance). This is an explicit project-only continuity heuristic,
not a physically validated motion limit. The cache was rebuilt from the
existing local GRIB; the source GRIB and 2011–2019 IMDAA reference were not
modified.

| Diagnostic | Rebuilt replay |
|---|---:|
| Track records | 188 |
| Geographic detections retained | 328 |
| Consecutive within-track links | 140 |
| Median link distance | 113.60 km |
| Maximum link distance | 317.13 km |
| Links over 500 km | 0 |
| Links over 1,000 km | 0 |
| Maximum implied link speed | 52.86 km/h |
| Links spanning 12 hours (one missing 6-hour step) | 43 |

The tighter gate fragments detections into many short tracks (median one point;
longest 25 points). It removes the very long false links but does not establish
that any resulting track is a meteorologically correct storm trajectory.
Where the allowed one-step gap occurs, the map still draws a line between the
two actual detections; that segment is not an observed/interpolated point.

## Map overlays

- The white polyline is assembled from the dominant track's geographic
  `track_points`, in chronological order, up to the selected valid time. It
  uses actual latitude/longitude and the same longitude-x / latitude-y map
  projection as the geographic grid.
- The yellow dashed rectangle previously preferred the full track's union
  bounds, even while a single forecast timestep was selected. It now uses only
  that timestep's `selected_track_point.geographic_bbox`.
- Each marker is a spherical centroid of the thresholded GNN connected
  component. The recorded confidence and anomaly are component means, not
  values sampled at that centroid. Comparing those means with the nearest grid
  cell gave mean absolute differences of 0.05615 for GNN probability and
  0.04066 for anomaly score; maxima were 0.92098 and 0.46617, respectively.
  These discrepancies are consistent with a component centroid not being a
  representative grid cell and mean values not being point samples.
- Component bounds can themselves be broad: the meridional extent has a median
  of 53.4 km and a maximum of 2,241.7 km; 51 of 328 point bounds exceed 500 km
  meridionally. These are bounds of thresholded model components, not a
  cyclone's observed extent.

The UI legend now labels these as an “experimental GNN cluster-centroid path”
and “active GNN cluster bounds.”

## Scope and limitations

- Real: the NCMRWF TIGGE forecast, its geographic grid and valid times, and the
  2011–2019 IMDAA temperature reference kept separate from the 2020 event.
- Experimental/derived: temperature anomaly proxy, weakly supervised GNN
  probabilities, thresholded components, centroid detections, continuity-
  gated links, and project-specific severity.
- Not available: an independently verified Amphan track for this replay,
  calibrated GNN confidence, forecast-skill verification, or official warning
  status. No accuracy or operational-forecast claim is made.

## Validation

- Focused backend tests: 10 passed.
- Full backend suite: 75 passed, 10 warnings.
- Frontend TypeScript check: passed.
- Frontend production build: passed. The existing toolchain emitted its
  chunk-size and configuration notices.
