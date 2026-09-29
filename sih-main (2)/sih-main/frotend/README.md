# Stormtrace Command

1. PRIMARY OBJECTIVE

Create a visually spectacular but highly professional dashboard that makes the STORMTRACE-GNN pipeline understandable within seconds.

The interface should feel like a combination of:

a modern meteorological operations center

a premium scientific visualization platform

an AI inference console

an emergency-response command dashboard

It must look compact, elegant, futuristic, scientific and extremely polished.

It should contain sophisticated animation, but animation must communicate information rather than become visual noise.

The overall experience should feel suitable for:

IMD / NDRF / meteorologists / researchers / emergency-response operators / SIH judges

Do NOT make it look like a gaming dashboard.

Do NOT make it look like an ordinary SaaS admin panel.

Do NOT make it look like a generic weather app.

2. IMPORTANT: USE THE PROVIDED BACKEND

First inspect the complete backend ZIP.

The frontend must connect to the actual FastAPI backend.

Backend base URL must be configurable through an environment variable, for example:

VITE_API_BASE_URL=http://localhost:8000

Never hardcode the production API URL.

Create a centralized API client such as:

src/lib/api.ts

and keep ALL backend calls there.

Use proper TypeScript types for every response.

Do not scatter fetch calls throughout components.

Create:

API service layer

typed API models

React hooks for API state

loading states

error states

retry states

polling logic

request cancellation where appropriate

3. ACTUAL BACKEND CAPABILITIES TO INTEGRATE

The current backend exposes these real functions.

Health

GET /health

Display backend/system health.

Expected structure includes:

status

version

Create a live connection indicator in the header:

SYSTEM ONLINE

or

BACKEND OFFLINE

Do not fake this indicator.

Run Pipeline

POST /pipeline/run

Request:

{
  "event_name": "live_run",
  "seed": null
}

The API returns a job_id.

The pipeline is asynchronous.

When the user clicks:

RUN LIVE ANALYSIS

start the job and immediately begin polling:

GET /pipeline/status/{job_id}

until status becomes:

queued

running

done

failed

The UI must visually represent this lifecycle.

4. SHOW THE REAL 8-STAGE PIPELINE

The backend progress messages follow:

generating ensemble + baseline

building / loading cached spherical mesh

projecting ensemble mean + computing EFI

training GNN + scoring timesteps

linking anomaly clusters into a 4D bounding box

U-Net baseline vs conditional diffusion

physics consistency check

severity-tiered alert + persistence

Build a beautiful Pipeline Execution Timeline.

Example:

01  DATA INGESTION
    Ensemble + climatology

02  SPHERICAL MESH
    Icosahedral projection

03  EFI ANALYSIS
    30-year baseline comparison

04  GNN SCAN
    Anomaly classification

05  4D TRACKING
    Temporal trajectory linking

06  DIFFUSION
    12 km → 5 km downscaling

07  PHYSICS CHECK
    Conservation consistency

08  ALERT ENGINE
    Centroid + severity

Each stage should animate as:

Pending → Running → Complete

Use subtle particle movement / progress pulses / connected-node animation.

When the backend sends progress strings, map them to these stages.

Do not fabricate progress.

5. HISTORICAL REPLAY MUST BE A FIRST-CLASS FEATURE

The blueprint explicitly recommends Historical Replay Mode as the default judge-facing demonstration.

Integrate:

GET /replay/events

This should populate the available replay cases dynamically.

Current backend includes:

cyclone_amphan

north_india_heatwave

Then allow the user to select:

Historical Replay

Cyclone Amphan
North India Heatwave

and launch:

POST /replay/{event_key}

Use the returned job_id.

Poll:

GET /pipeline/status/{job_id}

until completion.

Create a dedicated REPLAY MODE visual treatment.

Example top banner:

HISTORICAL REPLAY
Reproducible offline-safe inference

Important:

Do not claim that synthetic backend outputs are actual meteorological observations.

When the backend is using the included synthetic data implementation, clearly label the visualization as:

SIMULATED / SYNTHETIC DEMO DATA

Do not fabricate real-world latitude/longitude or observational accuracy that the backend does not actually provide.

The blueprint specifically notes that the backend MVP can use historical replay and synthetic/cached development paths while the real NEPS-G ingestion is added later.

6. MAIN DASHBOARD DESIGN

Create an application shell.

Left sidebar

Compact vertical navigation:

STORMTRACE

Navigation:

Command Center

Live Analysis

Historical Replay

Anomaly Tracking

Downscaling Lab

Physics Validation

Alert Center

Verification

API / System

Bottom:

Backend status

Model status

Data mode

Use clean icons.

Use a collapsible sidebar.

7. HEADER

Top header:

Left:

STORMTRACE-GNN

small subtitle:

EXTREME WEATHER INTELLIGENCE

Center:

Current operating mode:

LIVE
or
HISTORICAL REPLAY

Right:

backend connection

last analysis

system health

settings

theme toggle

Add a subtle animated network/mesh motif behind the header.

8. COMMAND CENTER

The landing dashboard should immediately show the most important information.

Use a responsive grid.

Primary center area

Large interactive geospatial visualization.

Right side

ACTIVE THREAT

Show:

event name

severity

confidence

peak value

EFI peak

physics score

5 km radius

centroid

Bottom

Compact analytical cards:

EFI PEAK

DIFFUSION PEAK

U-NET PEAK

AMPLITUDE GAIN

PHYSICS SCORE

TRACKED POINTS

FORECAST WINDOW

PROCESSING TIME

9. MAP / SPATIAL VISUALIZATION

Use MapLibre GL as specified in the blueprint.

The map must be one of the visual focal points of the application.

Features:

dark scientific basemap

smooth zoom

pan

terrain-aware appearance where appropriate

anomaly layer

severity layer

bounding box

tracked trajectory

centroid marker

5 km impact radius

animated trajectory line

pulse effect around current threat

expandable fullscreen mode

The map must support:

Severity

Low
Moderate
Severe

Use a clear, accessible severity visual system.

Do NOT use excessive glowing neon colors everywhere.

The map should look like a professional meteorological visualization.

10. IMPORTANT GEOSPATIAL DATA RULE

The backend currently returns fields such as:

centroid_row

centroid_col

4D bounding-box values

peak value

severity

radius

It does NOT currently provide a conventional latitude / longitude pair in the alert response.

Therefore:

DO NOT fabricate geographic coordinates.

Until backend geospatial coordinates are available, present the centroid as:

Mesh Coordinate

and visualize it using the available backend geometry where technically appropriate.

Create the frontend architecture so that future backend responses containing:

latitude
longitude

can automatically be used without redesigning the map.

11. STAGE 1 — ANOMALY TRACKING VIEW

Create a dedicated page called:

ANOMALY TRACKING

This page should visually explain the spherical GNN.

Show:

Left

Interactive spherical mesh visualization.

Render an animated icosahedral / spherical network.

Show:

mesh nodes

active anomaly nodes

propagation

highlighted clusters

Animate message passing between nodes.

Center

EFI ANALYSIS

Large numerical EFI indicator.

Show:

EFI
+0.82

with a mini distribution visualization.

Show explanatory text:

Extreme Forecast Index relative to climatological baseline

Do not invent scientific values; use backend results.

Right

4D TRAJECTORY

Animated timeline:

D+3 → D+4 → D+5 → ... → D+10

Show the anomaly bounding box evolving across time.

The blueprint explicitly describes this as a moving 4D object instead of a single-frame anomaly map.

12. FORECAST TIMELINE

Create a beautiful horizontal forecast timeline.

D+3   D+4   D+5   D+6   D+7   D+8   D+9   D+10
 │     │     │     │     │     │     │     │

Animate an anomaly indicator moving across it.

Interaction:

Click a forecast step → update the visualization.

Use smooth interpolation between steps.

Make it feel like temporal weather tracking rather than a normal slider.

13. STAGE 2 — DOWNSCALING LAB

Create a page:

AMPLITUDE-PRESERVING DOWNSCALING

This should be one of the strongest parts of the UI.

Create a three-panel comparison:

PANEL A

RAW / COARSE

12 km

PANEL B

U-NET BASELINE

12 km → 5 km

PANEL C

STORMTRACE DIFFUSION

12 km → 5 km

Use synchronized visualization.

When the user moves the comparison slider:

all three fields should remain aligned.

Show the extreme peak visually.

14. SPECTRAL-SMOOTHING DEMONSTRATION

Create a visual section titled:

WHY DIFFUSION?

Explain:

Regression-based downscaling can smooth extreme peaks toward the mean. STORMTRACE uses conditional diffusion to preserve extreme-value structure.

This comes directly from the project's stated technical rationale.

Create a visual metric:

U-NET PEAK
███████████

DIFFUSION PEAK
████████████████

Then show:

AMPLITUDE GAIN +XX.X%

using the actual backend value:

amplitude_gain_pct

Do not hardcode the percentage.

15. PHYSICS VALIDATION PANEL

Create:

PHYSICS CONSISTENCY

Large circular score:

0.87

Label:

Physics consistency score

Show:

ensemble agreement

conservation consistency

physics status

model confidence

The backend combines a post-hoc ensemble agreement score with a moisture-convergence consistency score. This should be reflected in the UI without pretending the system implements a complete fluid-dynamics solver.

Use status states such as:

CONSISTENT

REVIEW

LOW CONFIDENCE

based on actual backend values.

Do not claim:

PHYSICALLY GUARANTEED

because the blueprint explicitly says the model is penalized rather than hard-constrained.

16. ALERT ENGINE

Create a dedicated:

ALERT CENTER

Consume:

GET /alerts/latest

Support:

latest alerts

severity filtering

event filtering

pagination/load-more behavior

sorting

Backend supports:

severity=low

severity=moderate

severity=severe

and:

event_name

Use those actual query parameters.

17. ALERT CARDS

Each alert card should display:

SEVERE

Cyclone Amphan
Alert #124

Peak        42.8
EFI         0.91
Confidence  0.86
Physics     0.89

Impact radius
5 km

Centroid
Mesh [row, col]

Generated
2026-...

Clicking an alert should open an expanded detail drawer/modal.

Fetch the authoritative record using:

GET /alerts/{alert_id}

Do not rely solely on stale list data.

18. SEVERITY DESIGN

Use exactly the backend categories:

low

moderate

severe

Do not create extra severity classifications that the backend does not support.

The backend combines:

normalized amplitude

EFI

physics consistency

and can downgrade a severe call when the physics confidence is insufficient. Reflect this relationship in the UI.

19. VERIFICATION DASHBOARD

Create:

MODEL VERIFICATION

Consume:

GET /metrics

Show:

Total Alerts

total_alerts

Severity Distribution

by_severity.low

by_severity.moderate

by_severity.severe

Average Physics Score

avg_physics_score

Average Diffusion vs U-Net Amplitude Gain

avg_diffusion_vs_unet_amplitude_gain_pct

Use animated counters and professional charts.

Do not fabricate accuracy percentages.

The blueprint explicitly instructs that final evaluation should use event-based holdouts and report tracking recall/false-alarm rate and peak-amplitude retention when those metrics are actually evaluated.

Since the current API's /metrics does not expose tracking recall/false-alarm rate, do not invent them.

20. API / SYSTEM PAGE

Create a developer-oriented page:

SYSTEM / API

Show:

Backend URL

Connection status

API version

Recent jobs

Current job

Last job duration

Endpoints available

Add a compact API inspector.

When a pipeline job completes, display the actual JSON response in a collapsible terminal-style panel.

Example:

{
  "alert_id": 12,
  "severity": "moderate",
  "confidence": 0.82,
  ...
}

Add:

COPY JSON

button.

Do not expose unnecessary stack traces to normal users, but provide a developer-mode expandable error panel.

21. PIPELINE JOB EXPERIENCE

When a user starts an analysis:

Immediately open an elegant full-width execution panel.

Display:

STORMTRACE ENGINE
────────────────────────────────────

INITIALIZING

✓ Ensemble + baseline
✓ Spherical mesh
● EFI computation
○ GNN anomaly scoring
○ 4D trajectory
○ Diffusion downscaling
○ Physics validation
○ Alert generation

Use:

progress pulse

particle movement

subtle scanning line

animated node connections

progress percentage where derivable

Do not fabricate percentage values if the backend only provides stage text.

Instead show:

STEP 3 OF 8

This is more accurate.

22. FAILED JOB HANDLING

When:

status = failed

show:

PIPELINE FAILED

with:

friendly summary

job ID

backend error message

retry button

Do not crash the interface.

Do not leave a spinner running forever.

Set sensible polling limits and stop polling once:

done

failed

23. LIVE VS REPLAY

Make this extremely clear.

Top-level mode switch:

[ LIVE ANALYSIS ] [ HISTORICAL REPLAY ]

Live:

Real-time pipeline execution

Replay:

Deterministic offline-safe demonstration

When running replay, display the selected event prominently.

24. JUDGE DEMO MODE

Add a dedicated button:

LAUNCH DEMO

This should make the SIH presentation extremely smooth.

Flow:

Open Historical Replay

Select event

Run replay

Pipeline animation begins

Stage 1 GNN visualization appears

EFI grows

4D bounding box tracks anomaly

Diffusion/U-Net comparison appears

Physics check appears

Centroid + severity appears

Alert JSON appears

Verification metrics appear

The blueprint explicitly recommends exactly this kind of narrative for the judge-facing demonstration.

Add:

AUTO PLAY DEMO

which advances the UI through the stages after the real backend job provides the required output.

Do not fabricate backend calculations just for animation.

25. ANIMATIONS

Use Framer Motion or the current best React animation library available in the project.

Animations should include:

Page transitions

20–300 ms subtle transitions.

Cards

Soft entrance animations.

Metric counters

Count from 0 to actual value.

Map

Animated threat pulse.

GNN

Node activation / message-passing animation.

Timeline

Moving threat indicator.

Diffusion

Progressive noise-to-structure visualization.

Physics

Animated circular validation indicator.

Alerts

Severity-aware pulse.

Sidebar

Smooth collapse.

Avoid:

excessive bouncing

cartoon animations

huge transitions

distracting background videos

constant motion everywhere

The interface should feel expensive and controlled.

26. CSS / VISUAL DESIGN

The CSS quality must be exceptional.

Use:

Tailwind CSS + carefully authored custom CSS

Do not rely solely on default Tailwind components.

Create a coherent design system.

Suggested aesthetic:

Background

Very dark navy / near-black scientific interface.

Panels

Dark translucent surfaces.

Subtle borders.

Minimal glassmorphism.

Typography

Use a highly readable modern sans-serif.

Large numeric typography for scientific metrics.

Monospaced font for:

API payloads

job IDs

coordinates

system logs

Borders

1px low-contrast borders.

Soft inner highlights.

Shadows

Very subtle.

Never excessive neon glow.

Cards

Compact.

High information density.

Clean hierarchy.

Radius

Moderately rounded, not bubble-like.

27. VISUAL LANGUAGE

Create a visual identity around:

spherical mesh

atmospheric particles

topographic contours

pressure/isobar motifs

data grids

waveforms

trajectory traces

radar-like sweeps

These should be subtle background elements.

Do not let decoration interfere with data.

28. DARK/LIGHT MODE

Default:

Dark Scientific Mode

Also support a polished light mode.

Persist preference.

Make sure charts, maps and metrics remain readable in both.

29. RESPONSIVE DESIGN

Desktop is the primary environment.

But make the dashboard responsive.

Support:

1440px

1920px

laptop

tablet

At smaller widths, intelligently rearrange:

Map

↓

Threat metrics

↓

Timeline

↓

Physics

↓

Alerts

Do not simply shrink everything.

30. ACCESSIBILITY

Implement:

keyboard navigation

visible focus states

semantic buttons

ARIA labels

sufficient contrast

reduced motion support

When prefers-reduced-motion is enabled, replace elaborate animations with static transitions.

31. COMPONENT ARCHITECTURE

Create a clean architecture.

Example:

src/
├── components/
│   ├── layout/
│   ├── map/
│   ├── pipeline/
│   ├── tracking/
│   ├── diffusion/
│   ├── physics/
│   ├── alerts/
│   ├── metrics/
│   └── ui/
│
├── pages/
│   ├── CommandCenter.tsx
│   ├── LiveAnalysis.tsx
│   ├── HistoricalReplay.tsx
│   ├── AnomalyTracking.tsx
│   ├── DownscalingLab.tsx
│   ├── PhysicsValidation.tsx
│   ├── AlertCenter.tsx
│   ├── Verification.tsx
│   └── System.tsx
│
├── hooks/
│   ├── usePipelineJob.ts
│   ├── useAlerts.ts
│   ├── useMetrics.ts
│   └── useBackendHealth.ts
│
├── lib/
│   ├── api.ts
│   ├── types.ts
│   └── utils.ts
│
└── styles/

Keep components reusable.

Avoid giant monolithic components.

32. STATE MANAGEMENT

Use a lightweight modern React approach.

React Query / TanStack Query is preferred for:

API caching

polling

retries

stale data

loading state

query invalidation

Use local React state for UI interactions when sufficient.

33. TYPESCRIPT REQUIREMENTS

Strict TypeScript.

Create types corresponding to the actual backend.

At minimum represent:

type JobStatus =
  | "queued"
  | "running"
  | "done"
  | "failed";

Create interfaces for:

PipelineRunRequest

PipelineJobStatus

Alert

BoundingBox4D

Metrics

ReplayEvents

PipelineResult

The types must match the backend, not assumptions.

34. API CONTRACT

Implement these calls:

GET  /health

POST /pipeline/run

GET  /pipeline/status/{job_id}

POST /pipeline/run_sync

POST /replay/{event_key}

GET  /replay/events

GET  /alerts/latest

GET  /alerts/{alert_id}

GET  /metrics

Every one of these backend capabilities should be accessible from an appropriate part of the dashboard.

Do not leave any backend capability disconnected.

35. SYNC PIPELINE

POST /pipeline/run_sync

This is intentionally blocking and intended for local testing.

Put it under:

Developer / Advanced

with a warning:

Blocking execution — not recommended for normal dashboard operation.

Use the async pipeline as the primary workflow.

This corresponds directly to the backend's current architecture.

36. EMPTY STATES

Design intelligent empty states.

Example:

No alerts yet

No anomaly alerts have been persisted.
Run an analysis or launch Historical Replay.

Button:

RUN ANALYSIS

37. ERROR STATES

Examples:

Backend unavailable:

STORMTRACE ENGINE UNREACHABLE

with:

RETRY CONNECTION

No job:

JOB NOT FOUND

API failure:

REQUEST FAILED

Always show actionable recovery.

38. PERFORMANCE

The frontend should remain fast.

Use:

lazy-loaded pages

memoized map layers

debounced controls

virtualized long lists if required

limited animation work

efficient polling

cleanup of intervals on unmount

Polling should NOT continue once a job completes/fails.

39. DO NOT FAKE DATA

This is extremely important.

Never create fake API responses to make the UI appear functional.

Do not hardcode:

severity values

EFI scores

confidence

amplitude gain

alert counts

physics scores

backend health

processing time

Use the actual backend.

For purely visual placeholders, clearly label them:

DEMO VISUALIZATION

or

SIMULATED DATA

Do not confuse simulated values with real inference output.

40. DATA MODEL VISUALIZATION

When the backend result includes:

bounding_box
n_tracks
gnn_loss_history
seed
elapsed_seconds
efi_peak
unet_peak
diffusion_peak
amplitude_gain_pct
physics_score

visualize them elegantly.

GNN loss history

Create a small line chart.

Bounding box

Display dimensions.

Tracks

Display number of tracks.

Seed

Show under technical provenance.

Runtime

Show pipeline runtime.

41. PROVENANCE

Every result should have a small expandable:

PROVENANCE

panel containing:

event name

seed

mode

generated time

model stages

backend version

job ID

source mode

This makes the application feel scientifically auditable.

42. SCIENTIFIC HONESTY

The frontend must reflect the project accurately.

The backend currently uses synthetic multivariable ensemble data as a stand-in for NEPS-G/ERA5.

Therefore:

Do not write:

"Live NEPS-G operational forecast"

unless the backend actually provides it.

Instead show:

DATA MODE: SYNTHETIC / DEMO

for the current implementation.

The blueprint describes real NEPS-G, NCUM, ERA5/IMDAA as the intended datasets, while also explicitly providing a development/MVP route around cached/synthetic data.

43. FUTURE-PROOF THE FRONTEND

Architect the frontend so that later the backend can provide:

real lat/lon

real NetCDF/GRIB ingestion

multiple variables

ensemble members

true 5 km raster outputs

multiple simultaneous tracks

real meteorological layers

learned severity confidence intervals

without redesigning the whole application.

Create adapters rather than tightly coupling visualization components to current synthetic response structure.

44. LANDING SCREEN

The initial Command Center should immediately answer:

WHAT?

Extreme weather anomaly detected.

WHERE?

Centroid / tracked region.

WHEN?

D+3 → D+10 temporal window.

HOW SEVERE?

Low / Moderate / Severe.

HOW CONFIDENT?

Confidence score.

WHY?

EFI + ensemble agreement + physics consistency.

WHAT CHANGED?

Diffusion vs U-Net amplitude preservation.

45. MICRO-INTERACTIONS

Add premium interactions:

Hover over EFI:

show explanatory tooltip.

Hover over Physics:

show what contributes to score.

Hover over Diffusion Gain:

show U-Net peak vs diffusion peak.

Hover over timeline:

show forecast step.

Click severity:

filter alert center.

Click track:

focus map.

Click alert:

open full detail.

Click map centroid:

open threat detail.

46. SEARCH / FILTERING

Alert center should allow:

Search event name.

Severity filter.

Sort newest/oldest.

Selected event filter.

Do not create backend filters that do not exist.

Frontend-only filtering is acceptable for already loaded results, but make API-backed filtering primary when the backend supports it.

47. TOASTS

Use compact professional notifications:

ANALYSIS STARTED

REPLAY STARTED

PIPELINE COMPLETE

ALERT PERSISTED

BACKEND CONNECTION RESTORED

PIPELINE FAILED

Do not flood the user.

48. DEMO-FIRST DETAILS

I want the application to be exceptionally effective during a presentation.

Make these elements visually dominant:

GNN tracking

EFI

4D trajectory

Diffusion vs U-Net

Physics validation

5 km alert radius

Severity

Verification

The core differentiator of the project is the combination of spherical/mesh-native anomaly tracking, ensemble-native forecasting and generative downscaling focused on preserving extremes rather than simply minimizing average error.

49. VISUAL STORYTELLING

The dashboard should communicate this pipeline visually:

GLOBAL EPS
   ↓
SPHERICAL MESH
   ↓
EFI
   ↓
GNN
   ↓
ANOMALY TRACK
   ↓
4D BOUNDING BOX
   ↓
12 km CROP
   ↓
U-NET vs DIFFUSION
   ↓
PHYSICS CHECK
   ↓
5 km IMPACT FIELD
   ↓
CENTROID
   ↓
SEVERITY
   ↓
ALERT API

Turn this into an interactive animated process visualization somewhere in the application.

50. FINAL VISUAL QUALITY BAR

The final product should look like an actual advanced scientific command platform.

Think:

NASA-style scientific visualization + premium AI operations console + modern meteorological center

but with its own identity.

Use:

elegant spacing

excellent typography

restrained gradients

subtle atmospheric backgrounds

high-quality data visualization

polished transitions

crisp icons

sophisticated map styling

compact cards

strong information hierarchy

Every screen should feel intentional.

No default-template appearance.

No generic dashboard cards.

No lorem ipsum.

No unnecessary giant headings.

No fake charts.

No fake weather statistics.

51. ACCEPTANCE CRITERIA

Do not consider the project complete until all of the following work:

Backend connectivity

health endpoint works

connection indicator reflects actual backend state

Pipeline

user can start /pipeline/run

job ID is captured

status is polled

progress is visualized

completion result is rendered

failure is handled

Replay

replay events are loaded from /replay/events

user can select event

replay starts from /replay/{event_key}

replay status is tracked

replay result is rendered

Alerts

latest alerts load

severity filtering works

event filtering works

individual alert detail loads

Metrics

/metrics loads

metrics update after a completed pipeline

charts use real API data

Developer mode

synchronous endpoint can be invoked intentionally

actual JSON can be inspected

UI

desktop polished

responsive

dark mode polished

animations smooth

no broken interactions

no console errors

no fake functionality

52. FINAL IMPLEMENTATION INSTRUCTION

Before writing the frontend:

Inspect the supplied backend ZIP completely.

Inspect the supplied STORMTRACE-GNN blueprint PDF completely.

Derive the API contracts from the actual backend.

Build the frontend around those actual contracts.

Preserve the project's scientific terminology.

Do not invent unsupported backend capabilities.

Use mock/demo visualization only where the current API genuinely does not expose enough data, and label it clearly.

Keep the application modular so real NEPS-G/NetCDF/GRIB data can be connected later.

Make the Historical Replay flow the polished default demonstration.

Finish with a fully usable, visually refined dashboard rather than a static UI mockup.

Do not stop at designing the interface. Implement the complete working frontend and connect every currently available backend function to the UI.

The final result must be a functional STORMTRACE-GNN command dashboard, not a landing page.

One additional instruction I would append to Lovable

Before finalizing, run through every API interaction manually in the UI: health → replay events → replay → job polling → completed result → alerts → alert detail → metrics → live pipeline → developer sync pipeline. Fix every broken interaction, TypeScript error, console error, loading-state issue, polling leak, and CORS/API configuration issue you encounter.

That last paragraph is important because your backend is already structured around asynchronous jobs and persistent alert records, so a beautiful frontend that only displays static cards would waste most of the work already present in the ZIP. The backend README confirms that the implementation is intended to be used exactly this way.

This project was built with [Lovable](https://lovable.dev).

**Live app**: https://stormtrace-ai-command.lovable.app

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/383deedc-2bf7-4346-bcb6-afcafbfeac2b).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
