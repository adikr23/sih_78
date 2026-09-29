# STORMTRACE-GNN backend

STORMTRACE-GNN includes a FastAPI service and an experimental weather analysis
pipeline. Real-data access is conditional on user-provided ERA5/CDS credentials
and local official NEPS-G forecast files. The implementation is **not an
operational forecast service** and its downscaling output is not verified
against a genuine high-resolution reference.

## Data modes

Set `DATA_MODE` explicitly:

| Mode | Forecast input | EFI reference | Synthetic allowed |
|---|---|---|---|
| `REAL` | Local official NEPS-G forecast file (`NEPS_G_FORECAST_PATH`; this project does not claim a live NEPS-G API) | Local ERA5 climatology or ERA5 downloaded using CDS | No |
| `LOCAL_REAL` | Local official NEPS-G forecast file | Local ERA5 climatology file only | No |
| `SYNTHETIC_DEMO` | Explicit synthetic demonstration provider | Synthetic demonstration climatology | Yes |

In `REAL`, both the NEPS-G forecast and real ERA5 reference are required. A CDS
account does not make data available until a retrieval succeeds and the
downloaded file opens and passes validation. In `LOCAL_REAL`, both local file
paths must be present. A missing or invalid real source raises a clear error;
it is never substituted with synthetic data.

The API identifies the configured mode in health, analysis results, and system
status. `public_weather` is a backwards-compatible provider selector whose
current data is synthetic; use `SYNTHETIC_DEMO` mode and treat it as demo-only.

## Current forecast state and live pipeline

The local-real pipeline is exposed through the generalized `/api/v1` routes.
`LOCAL_REAL` validates the configured TIGGE GRIB through the existing grouped
cfgrib adapter and serves the prepared scientific cache without reparsing it
for every browser request. Set `FORECAST_AUTO_INGEST=true` to run the
non-blocking local-cycle validator periodically; configure the interval with
`FORECAST_INTERVAL_MINUTES`. A missing or invalid source reports
`LIVE INGESTION UNAVAILABLE` and never falls back to synthetic data.

Useful routes are:

- `/api/v1/forecast/status` — current cycle, cache, freshness, and provenance state
- `/api/v1/forecast/latest` — current cached forecast metadata
- `/api/v1/forecast/timeline` — actual valid forecast times
- `/api/v1/forecast/field` — downsampled visualization fields
- `/api/v1/tracking/current` and `/api/v1/alerts/current` — derived current products

The checked-in Amphan cache is a historical local TIGGE cycle. It is therefore
reported as `STALE` relative to the current wall clock until a newer source
cycle is ingested. Remote automatic NCMRWF retrieval is not implemented
without an approved provider endpoint and credentials; `REAL` reports that
limitation explicitly rather than presenting the historical cache as current.

## Installation and startup (Windows PowerShell)

From this backend directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m src.cli system-check
python -m uvicorn src.main:app --reload --port 8000
```

The API creates/initializes its configured database on startup. SQLite is the
development default. For a shared job registry across multiple workers, set
`JOB_STORE_BACKEND=redis` and configure a reachable `REDIS_URL`; the
process-local fallback is only suitable for single-process development.

## Environment configuration

Values can be set in the backend `.env` file or the process environment. Do not
put CDS credentials in frontend configuration, source code, or committed files.
The backend also reads the backend user's standard `~/.cdsapirc` `url` and
`key` entries when `CDS_UID`/`CDS_API_KEY` are not set.

| Variable | Purpose |
|---|---|
| `DATA_MODE` | `REAL`, `LOCAL_REAL`, `TIGGE_AUTO`, or `SYNTHETIC_DEMO` |
| `TIGGE_AUTO_ENABLED` | Enable automated NCMRWF TIGGE retrieval through the configured CDS/ECDS client |
| `TIGGE_LOOKBACK_CYCLES` | Number of recent 00/12 UTC cycles to try |
| `TIGGE_MAX_LEAD_HOURS` | Maximum requested lead time, default 240 |
| `TIGGE_REGION` | Requested region as `north,west,south,east` |
| `TIGGE_CACHE_DIR` | Atomic download cache directory |
| `DATA_PROVIDER` | Provider selector; normally leave at `public_weather` for backwards compatibility or use `synthetic_demo` for demos |
| `DATABASE_URL` | SQLAlchemy database URL; development default is SQLite |
| `JOB_STORE_BACKEND` | `memory` for local single-process use or `redis` |
| `REDIS_URL` | Redis connection string; never returned by system-status APIs |
| `CDS_URL` | CDS API HTTPS URL |
| `CDS_UID`, `CDS_API_KEY` | CDS credentials, backend environment only |
| `ERA5_DATASET` | Currently supported dataset: `reanalysis-era5-single-levels` |
| `ERA5_VARIABLES` | Comma-separated official CDS variable names |
| `ERA5_DOWNLOAD_AREA` | `north,west,south,east` degrees |
| `ERA5_RESOLUTION` | `latitude_step/longitude_step`, in degrees |
| `ERA5_CACHE_DIR` | Download cache directory; created as needed |
| `ERA5_START_DATE`, `ERA5_END_DATE` | Explicit `YYYY-MM-DD` reference period used by the optional CDS climatology download |
| `ERA5_CLIMATOLOGY_PATH` | Local ERA5 reference NetCDF, GRIB/GRIB2, Zarr, or supported NPZ file |
| `ERA5_FORECAST_PATH` | Optional local ERA5 time series when using the ERA5 provider directly |
| `NEPS_G_FORECAST_PATH` | Local official NEPS-G forecast file |
| `NEPS_G_ACTIVE` | Indicates an expected NEPS-G source; does not create or download data |
| `FORECAST_AUTO_INGEST` | Enable the non-blocking local forecast-cycle validator |
| `FORECAST_INTERVAL_MINUTES` | Interval between automatic local-cycle checks |
| `FORECAST_STALE_AFTER_MINUTES` | Configured freshness threshold for future cycles |
| `DATA_VARIABLES` | Optional comma-separated source variable names, in canonical order `t2m,rainfall,wind_u,wind_v` |
| `FORECAST_INITIALIZATION_TIME` | Select an initialization when a local forecast has multiple initialization times |
| `FORECAST_LEAD_HOURS` | Optional forecast lead selection |
| `DATA_DOMAIN` | Optional local crop as `south,north,west,east` |
| `HIGH_RESOLUTION_TARGET_PATH` | Optional candidate target file; its resolution/validity is not assumed from its path |
| `VITE_API_BASE_URL` | Frontend API base URL; configure in the frontend environment, not with CDS credentials |

ERA5 defaults request 2 m temperature, total precipitation, and 10 m wind
components. Those fields do not contain specific humidity `q`. The pipeline
does not synthesize `q`; where `q/u/v` are not all genuinely present, physics
diagnostics must identify use of the existing proxy rather than claim
moisture-flux convergence from unavailable inputs.

## CDS credentials and one-off download

Set the credentials in the backend process environment (or an untracked
backend `.env`):

```powershell
$env:CDS_URL = "https://cds.climate.copernicus.eu/api"
$env:CDS_UID = "<your CDS user identifier>"
$env:CDS_API_KEY = "<your CDS API key>"
```

Use the values and URL currently specified by the CDS account. The application
does not print or return the credential values. To request data, configure the
dataset, variables, region, resolution, and exact date period, then invoke the
Python downloader:

```powershell
$env:ERA5_START_DATE = "2024-01-01"
$env:ERA5_END_DATE = "2024-01-02"
$env:ERA5_DOWNLOAD_AREA = "10,20,0,30"
python -c "from src.era5_cds import Era5CDSDownloader; print(Era5CDSDownloader().download_climatology())"
```

The downloader validates credentials/configuration, divides multi-month
requests into monthly CDS requests, stores a NetCDF and JSON provenance
sidecar under `ERA5_CACHE_DIR`, and reuses a cache entry only when its
provenance sidecar matches the request. Do not use a short weather-event
sample as a climatology without assessing temporal/seasonal representativeness.
CDS access, authentication, quotas, and product availability depend on the
account and are not asserted by local configuration checks.

## Local NEPS-G and ERA5 input schema

The application does not invent or call a NEPS-G API endpoint. The default
`NEPS_G_FORECAST_PATH` is
`data/neps_g/tigge_ncmrwf_amphan_20200516_00.grib`; override it to use another
authorized local forecast. `DATA_MODE=REAL` and `LOCAL_REAL` fail clearly if
the configured GRIB is missing. Real mode also requires an ERA5 reference
source (local `ERA5_CLIMATOLOGY_PATH` or the configured CDS retrieval); the
forecast and reference grids/units must match for EFI.

NCMRWF TIGGE GRIB is read with `cfgrib.open_datasets()` because level-specific
fields occur in separate GRIB groups. The adapter requires `t2m`, `tp`, `u10`,
and `v10` from centre `dems`, data type `pf`, GRIB edition 2. It assembles
canonical `(steps, members, variables, latitude, longitude)` output while
preserving the ensemble member identifiers and GRIB valid-time coordinates.
`tp` is an accumulation (`stepType=accum`), not an instantaneous rate. It is
converted from `kg m-2` to accumulated water-equivalent depth in metres using
the explicit factor `0.001` so it can be compared with ERA5's precipitation
depth units; no division by forecast duration or rainfall-rate conversion is
performed. This operation and the accumulated-amount meaning are recorded in
provenance. Original GRIB values are not modified.

NetCDF and GRIB/GRIB2 use xarray (GRIB via `cfgrib`); Zarr directories must end
in `.zarr`. A local NEPS-G forecast must provide:

- one-dimensional `latitude`/`lat` and `longitude`/`lon` coordinates;
- a forecast-step/time dimension (`step`, `forecast_step`, `lead_time`, or
  `time`) and a member dimension (`number`, `ensemble_member`, `member`, or
  `realization`);
- four variables matching canonical order: `t2m`, `rainfall`, `wind_u`,
  `wind_v`, or their supported source aliases (for example `2m_temperature`,
  `total_precipitation`, `10m_u_component_of_wind`,
  `10m_v_component_of_wind`);
- each variable on exactly the forecast-step, member, latitude, and longitude
  dimensions. Optional `valid_time`, forecast initialization, units, and lead
  metadata are preserved when provided.

Local ERA5 reference data must provide time/sample, latitude, longitude
coordinates and the corresponding four canonical variables/source aliases.
The normalized shapes are `(steps, members, variables, lat, lon)` for a
forecast and `(samples, variables, lat, lon)` for the reference. A deterministic
ERA5 time series can be loaded by the direct ERA5 provider with a singleton
member dimension; it is not represented as a multi-member ensemble.

The ingestion QC rejects malformed dimensions, non-finite/missing fields,
duplicate or unordered coordinates, invalid geographic bounds, and supported
physically implausible values. It does not silently drop bad samples or
rearrange data. QC results and source variable names are attached to
provenance. Units are preserved; ensure forecast and reference units are
compatible before calculating EFI.

## How to Run the Amphan Historical Replay

The demo uses the precomputed local IMDAA cache. It does not load ZIP archives
in the browser and does not retrain models when **Run Amphan Replay** is clicked.
If the processed cache is missing, prepare it once from the backend directory
before starting the demo:

```powershell
cd "C:\Users\adikr\Downloads\sih-main (2)\sih-main (2)\sih-main\backend\stormtrace-gnn-backend"
.\.venv\Scripts\Activate.ps1
python scripts\build_imdaa_cache.py
```

Start the backend in one PowerShell window:

```powershell
cd "C:\Users\adikr\Downloads\sih-main (2)\sih-main (2)\sih-main\backend\stormtrace-gnn-backend"
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Start the frontend in a second PowerShell window:

```powershell
cd "C:\Users\adikr\Downloads\sih-main (2)\sih-main (2)\sih-main\frotend"
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173/historical-replay` and click **Run Amphan Replay**.
The page loads cached event results, fields, tracks, alert, and model previews
from the backend API. The field API can return explicitly marked
visualization-downsampled arrays; the underlying cached scientific arrays are
unchanged.

Data and scientific scope:

- IMDAA source archives: `backend\stormtrace-gnn-backend\data\imdaa\`.
- Processed replay cache: `backend\stormtrace-gnn-backend\data\processed\`.
- Reference period: May 2011–2019 only; the 2020 event is held out.
- Event period: Cyclone Amphan, May 16–21, 2020; 24 six-hourly timestamps.
- This is an IMDAA historical replay, **not a live forecast**. NEPS-G is not
  connected.
- Rainfall APCP is preserved as raw data but disabled in scientific analysis;
  units and accumulation semantics remain unresolved.
- Specific humidity `q` is unavailable, so MFC is unavailable. Physics results
  remain labeled `PHYSICS_PROXY`.
- Independent high-resolution verification is unavailable. U-Net and diffusion
  outputs are experimental reconstructions, not validated 5 km observations.

## IMDAA Amphan historical replay

The optional historical replay reads the local IMDAA May ZIP archives directly
from `data/imdaa/`; it does not extract or rewrite source archives. It builds  a
regional cache for 5–30°N, 75–100°E, using May 2011–2019 as the temperature
reference and 2020-05-16 through 2020-05-21 as the held-out Amphan case. The
cache builder rejects missing timestamps, inconsistent grids, masked/non-finite
values, invalid physical levels, and non-identical duplicate wind records.
Identical U/V duplicate records are reduced to one logical sample and that
decision is preserved in provenance.

From the backend directory, prepare the offline replay:

```powershell
.\.venv\Scripts\python scripts\build_imdaa_cache.py
```

This creates compressed inputs and precomputed outputs in `data/processed/`.
It computes location-specific t2m empirical EFI-like scores against May
2011–2019 reference samples at matching synoptic hours, projects them to the
existing regional icosahedral mesh, trains a small GNN using weak labels derived
from the configured EFI threshold, links anomaly tracks, and caches the
experimental U-Net/diffusion outputs. The frontend replay button loads those
cached results; it does not retrain the models. Re-run the builder after
changing source archives or replay settings.

Settings:

| Variable | Default | Purpose |
|---|---|---|
| `IMDAA_ARCHIVE_DIR` | `data/imdaa` | Directory containing `2011.zip` through `2020.zip` |
| `IMDAA_PROCESSED_DIR` | `data/processed` | Generated reference, event, and replay caches |
| `AMPHAN_EFI_THRESHOLD` | `0.8` | Weak-label EFI threshold |
| `AMPHAN_GNN_THRESHOLD` | `0.5` | GNN probability threshold for tracking |
| `IMDAA_RAINFALL_MODE` | `disabled` | `disabled` hides rainfall fields; `raw_unverified` permits raw APCP inspection only |

APCP is retained with its filename and decoded-time provenance, but its local
NetCDF files do not specify units or accumulation interval. The decoded
coordinate remains one hour later than the filename time. Consequently,
rainfall is **disabled for EFI, downscaling, and verification** by default.
`IMDAA_RAINFALL_MODE=raw_unverified` only exposes the cached raw values and
does not validate their semantics or enable their use in calculations.

IMDAA is historical reanalysis, not a forecast ensemble or NEPS-G. EFI in this
replay uses t2m only. GNN training uses weak labels generated from the EFI
threshold and is not independently ground-truth trained. IMDAA has no
specific-humidity `q`, so the replay reports a `PHYSICS_PROXY` and does not
calculate moisture-flux convergence. Severity is labeled as prototype tiers,
not operational warning levels. Downscaled fields are **2x experimental
reconstructions / PROTOTYPE / PROXY TARGET**; no genuine approximately 5 km
target is present. Verification status is `NOT_AVAILABLE` without independent
observational labels or a real high-resolution reference.

Historical replay APIs:

```text
POST /api/amphan/replay
GET  /api/amphan/overview
GET  /api/amphan/timeline
GET  /api/amphan/field?variable=t2m&time_index=0
GET  /api/amphan/field?variable=efi&time_index=0
GET  /api/amphan/field?variable=unet
GET  /api/amphan/field?variable=diffusion
GET  /api/amphan/tracks
GET  /api/amphan/alert
GET  /api/amphan/verification
```

`POST /api/amphan/replay` only loads the prepared result cache. Start the
backend from this directory, then run the existing frontend from `frotend/`
with `VITE_API_BASE_URL` set to the backend URL. The Historical Replay page
shows the IMDAA source and mode, event timeline, geographic t2m/EFI fields,
track and centroid when detected, alert, downscaling proxy, physics proxy, and
verification/data limitations. If caches have not been built, the API reports
that the builder command is required.

## NCMRWF TIGGE Amphan forecast replay

The TIGGE forecast replay is a separate product from the IMDAA historical
reanalysis replay. It reads the local NCMRWF GRIB once during cache creation,
preserves its 11 perturbed members and 41 valid times, and writes compact field
and inference caches under `data/processed/`. The replay uses the existing
IMDAA May 2011–2019 `t2m` reference cache only, checks that the stored sample
times are exactly the 1,116 May records from 2011–2019, and rejects any 2020
reference samples. It restricts the forecast to the shared 5.04°–30°N,
75°–99.96°E reference domain.

Build once from this backend directory after the IMDAA reference cache has been
prepared and the local GRIB exists:

```powershell
.\.venv\Scripts\python scripts\build_tigge_amphan_cache.py
```

Optional arguments are `--forecast <local-grib-path>`, `--cache-dir
<processed-cache-directory>`, and `--training-epochs <count>`. The model is
trained only by this explicit preparation command. API and dashboard requests
load precomputed cache files and do not parse the GRIB or retrain the model.

The temperature field is compared with location-specific, UTC-hour-matched
IMDAA temperature samples to produce an empirical EFI-like score. The current
IMDAA reference does not include wind climatology, so wind magnitude is shown
as a within-forecast spatial-percentile anomaly proxy, not climatological EFI.
The combined anomaly score and the weakly supervised GNN are experimental and
uncalibrated. Geographic track detections use a stated score threshold and
are anomaly clusters, not a verified cyclone-center track. Severity is a
deterministic project tier based on the share of grid cells above a proxy
anomaly threshold; it is neither an official warning category nor calibrated
meteorological severity.

GRIB `tp` is accumulated precipitation (`kg m-2`), explicitly represented in
the cache as accumulated water-equivalent depth in metres. It is excluded from
anomaly analysis and is never interpreted as a rainfall rate. Specific
humidity/MFC are unavailable, physics stays `PHYSICS_PROXY`, and independent
verification remains `NOT_AVAILABLE / EXPERIMENTAL`. No validated 12 km to
5 km downscaling is claimed.

TIGGE replay APIs:

```text
POST /api/tigge/amphan/replay
GET  /api/tigge/amphan/overview
GET  /api/tigge/amphan/field?variable=t2m&time_index=0&visualization=true
GET  /api/tigge/amphan/field?variable=efi&time_index=0&visualization=true
GET  /api/tigge/amphan/field?variable=gnn_probability&time_index=0&visualization=true
GET  /api/tigge/amphan/tracks?time_index=0
GET  /api/tigge/amphan/alert?time_index=0
GET  /api/tigge/amphan/verification
```

Open `/tigge-replay` in the frontend to start the cached real-data replay,
select a valid time/forecast lead, view the geographic fields and anomaly track,
and inspect forecast metadata and limitations. Both IMDAA and TIGGE historical
replay dashboards remain available as separate views.

## API quick checks

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/api/system/status
Invoke-RestMethod http://127.0.0.1:8000/api/system/data-sources
Invoke-RestMethod http://127.0.0.1:8000/api/amphan/overview
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/analysis/run `
  -ContentType "application/json" -Body '{"event_name":"live_run"}'
```

Use the returned `job_id` with `GET /api/analysis/{job_id}` (or the existing
`/pipeline/status/{job_id}` route). `/api/data/available` reports whether the
configured provider can be selected. `/api/data/qc` returns
`not_available` until a persisted QC report is exposed. System status never
returns CDS credentials or the Redis connection URL.

## System check and tests

```powershell
python -m src.cli system-check
.\.venv\Scripts\python -m pytest -q
```

The system check distinguishes `READY`, `NOT CONFIGURED`, `CONFIGURED BUT DATA
UNAVAILABLE`, `CONFIGURED BUT UNVERIFIED`, `INVALID`, and `DEMO ONLY`. It does
not make a CDS request or claim credentials are valid based solely on their
presence.

## Scientific and operational limitations

- No real CDS request or NEPS-G ingestion is claimed until a real file is
  retrieved/opened and the full pipeline is executed against it.
- Current model variable set does not include `q`; moisture flux convergence
  cannot be claimed from the default ERA5 single-level variables. Use the
  explicit physics proxy unless appropriate real fields are available.
- EFI is an empirical prototype. The supplied reference distribution must be
  spatially/unit compatible and season/lead-time representative; automatic
  seasonal/lead stratification is not implied.
- Verification values must only be reported against a legitimate reference;
  otherwise verification is unavailable.
- TIGGE accumulated precipitation values must not be interpreted as rates.
  Forecast/reference unit and temporal-accumulation compatibility must be
  assessed before interpreting precipitation EFI.
- Downscaling is classified **PROTOTYPE / PROXY TARGET** unless an actual
  approximately 5 km reference dataset is available and evaluated. Interpolated
  model output is not ground truth. Genuine validated 12 km → 5 km
  downscaling requires a genuine high-resolution target dataset.
- The system is not operationally validated. CDS credentials, official
  NEPS-G data access, Redis service, and high-resolution target datasets are
  external requirements.
