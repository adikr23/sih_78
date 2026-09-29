# Map performance audit

## Changes measured

Benchmarked the warm local API field route using FastAPI TestClient against the
existing 64×64 visualization response. These are local development
measurements, not production/network guarantees.

No pre-change timing was captured before modifying the implementation, so an
honest numeric before/after speedup cannot be stated. The measured data below
is the post-change cold/warm behavior.

| Request | HTTP time | Payload |
|---|---:|---:|
| First measured field request (cold process/cache) | 165.4 ms | 25,905 compressed bytes; 79,218 decoded JSON bytes |
| Subsequent warm request | 59.75 ms | 25,905 compressed bytes |
| Warm request with gzip accepted | 59.53 ms | gzip, 25,905 bytes |

The response retains the same geographic arrays, field values, units, and
metadata. ETag is returned for immutable cache products and conditional
requests receive `304 Not Modified`.

## Before/after architecture

Before: each React Query key was timestep-specific; changing the slider could
show a loading state while multiple requests resolved. Field handling also
reopened the compressed analysis NPZ and the SVG generated one rectangle per
rendered map cell.

After:

- Existing GRIB-derived NPZ/JSON products remain the scientific source.
- Loaded NPZ arrays and replay JSON metadata are process-cached and invalidated
  when file modification/size metadata changes.
- Field APIs expose ETags and short immutable-product caching, with gzip for
  JSON responses over 1 KB.
- The Command Center keeps the previous map field rendered during an
  uncached timestep lookup, debounces rapidly moving targets, and prefetches
  neighboring steps into React Query's bounded cache.
- The field raster is drawn to a mounted canvas. Geographic labels and
  track/alert overlays remain SVG. Map geometry changes only when field data
  changes; there is no forecast-field interpolation.

## Memory calculation

The stated source grid Float32 size is:

`41 × 209 × 195 × 4 = 6,683,820 bytes`

Exact calculation: 41 × 209 × 195 = 1,670,955 values; 4-byte Float32 arrays
require approximately 6.37 MiB for one variable. Four variables require about
25.5 MiB, before browser object overhead. The current API serves an explicitly
visualization-sampled 64×64 field and does not load the entire source forecast
into browser memory.

## Limitations / measurements not collected

- The backend cache retains derived analysis grids (209×139), not all original
  five-dimensional member arrays. Per-member browser loading is not supported.
- Browser automation did not provide a stable render-duration or heap
  measurement in this environment.
- Full network waterfall, cross-browser, cold browser cache, rapid-drag
  request count, and production deployment payload measurements remain
  unmeasured.
- Adjacent prefetch improves likely next/previous navigation but all 41 steps
  are not downloaded eagerly.
