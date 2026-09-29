"""
Central configuration for STORMTRACE-GNN backend.
All tunables are env-overridable so the same code runs identically
in dev (toy grid) and a heavier deployment (larger grid / mesh / ensemble)
with zero code changes -- only these numbers move.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    # --- storage ---
    database_url: str = Field(
        default="sqlite:///./stormtrace.db",
        alias="DATABASE_URL",
    )
    job_store_backend: str = Field(default="memory", alias="JOB_STORE_BACKEND")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    data_mode: str = Field(default="SYNTHETIC_DEMO", alias="DATA_MODE")
    tigge_auto_enabled: bool = Field(default=False, alias="TIGGE_AUTO_ENABLED")
    data_dir: str = "data"
    data_provider: str = Field(default="public_weather", alias="DATA_PROVIDER")
    neps_g_forecast_path: Optional[str] = Field(
        default="data/neps_g/tigge_ncmrwf_amphan_20200516_00.grib",
        alias="NEPS_G_FORECAST_PATH",
    )
    neps_g_active: bool = Field(default=False, alias="NEPS_G_ACTIVE")
    forecast_auto_ingest: bool = Field(default=False, alias="FORECAST_AUTO_INGEST")
    forecast_interval_minutes: int = Field(default=60, alias="FORECAST_INTERVAL_MINUTES")
    forecast_stale_after_minutes: int = Field(default=1440, alias="FORECAST_STALE_AFTER_MINUTES")
    forecast_state_path: str = Field(default="data/processed/forecast_state.json", alias="FORECAST_STATE_PATH")
    tigge_lookback_cycles: int = Field(default=4, alias="TIGGE_LOOKBACK_CYCLES")
    tigge_max_lead_hours: int = Field(default=240, alias="TIGGE_MAX_LEAD_HOURS")
    tigge_region: str = Field(default="30,70,5,105", alias="TIGGE_REGION")
    tigge_cache_dir: str = Field(default="data/neps_g/auto", alias="TIGGE_CACHE_DIR")
    tigge_origin: str = Field(default="ncmrwf", alias="TIGGE_ORIGIN")
    tigge_dataset: str = Field(default="tigge-global", alias="TIGGE_DATASET")
    era5_forecast_path: Optional[str] = Field(default=None, alias="ERA5_FORECAST_PATH")
    era5_climatology_path: Optional[str] = Field(default=None, alias="ERA5_CLIMATOLOGY_PATH")
    era5_cache_dir: str = Field(default="data/era5_cache", alias="ERA5_CACHE_DIR")
    imdaa_archive_dir: str = Field(default="data/imdaa", alias="IMDAA_ARCHIVE_DIR")
    imdaa_processed_dir: str = Field(default="data/processed", alias="IMDAA_PROCESSED_DIR")
    imdaa_rainfall_mode: str = Field(default="disabled", alias="IMDAA_RAINFALL_MODE")
    amphan_efi_threshold: float = Field(default=0.8, alias="AMPHAN_EFI_THRESHOLD")
    amphan_gnn_threshold: float = Field(default=0.5, alias="AMPHAN_GNN_THRESHOLD")
    era5_dataset: str = Field(default="reanalysis-era5-single-levels", alias="ERA5_DATASET")
    era5_variables: str = Field(default="2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,total_precipitation", alias="ERA5_VARIABLES")
    era5_download_area: str = Field(default="35,-20,0,80", alias="ERA5_DOWNLOAD_AREA")
    era5_resolution: str = Field(default="0.25/0.25", alias="ERA5_RESOLUTION")
    era5_start_date: Optional[str] = Field(default=None, alias="ERA5_START_DATE")
    era5_end_date: Optional[str] = Field(default=None, alias="ERA5_END_DATE")
    cds_uid: Optional[str] = Field(default=None, alias="CDS_UID")
    cds_api_key: Optional[str] = Field(default=None, alias="CDS_API_KEY")
    cds_url: str = Field(default="https://cds.climate.copernicus.eu/api/v2", alias="CDS_URL")
    cds_timeout_seconds: int = Field(default=300, alias="CDS_TIMEOUT_SECONDS")
    high_resolution_target_path: Optional[str] = Field(default=None, alias="HIGH_RESOLUTION_TARGET_PATH")
    source_dataset_name: Optional[str] = Field(default=None, alias="SOURCE_DATASET_NAME")
    forecast_initialization_time: Optional[str] = Field(default=None, alias="FORECAST_INITIALIZATION_TIME")
    forecast_lead_hours: Optional[float] = Field(default=None, alias="FORECAST_LEAD_HOURS")
    forecast_start_time: Optional[str] = Field(default=None, alias="FORECAST_START_TIME")
    forecast_end_time: Optional[str] = Field(default=None, alias="FORECAST_END_TIME")
    data_domain: Optional[str] = Field(default=None, alias="DATA_DOMAIN")
    data_spatial_resolution_km: Optional[float] = Field(default=None, alias="DATA_SPATIAL_RESOLUTION_KM")
    data_variables: Optional[str] = Field(default=None, alias="DATA_VARIABLES")
    ensemble_member: Optional[str] = Field(default=None, alias="ENSEMBLE_MEMBER")
    input_resolution_km: Optional[float] = Field(default=None, alias="INPUT_RESOLUTION_KM")
    target_resolution_km: Optional[float] = Field(default=None, alias="TARGET_RESOLUTION_KM")

    # --- synthetic ensemble geometry (stand-in for a public/local forecast source) ---
    grid_size: int = Field(default=32, alias="GRID_SIZE")
    n_ensemble_members: int = Field(default=8, alias="N_ENSEMBLE_MEMBERS")
    n_baseline_years: int = Field(default=30, alias="N_BASELINE_YEARS")
    # variables tracked per grid cell: t2m, rainfall_proxy, wind_u, wind_v
    n_variables: int = 4
    variable_names: tuple = ("t2m", "rainfall", "wind_u", "wind_v")

    # --- temporal window: D+3 .. D+10 -> 8 forecast steps ---
    n_forecast_steps: int = Field(default=8, alias="N_FORECAST_STEPS")

    # --- spherical mesh (icosahedral) ---
    mesh_subdivisions: int = Field(default=4, alias="MESH_SUBDIVISIONS")

    # --- GNN ---
    gnn_hidden: int = 32
    gnn_layers: int = 3
    gnn_epochs: int = 120
    gnn_lr: float = 5e-3

    # --- EFI ---
    efi_anomaly_threshold: float = 0.5  # >0.5 => flagged anomalous
    max_track_gap_steps: int = Field(default=0, alias="MAX_TRACK_GAP_STEPS")
    track_max_link_distance: float = Field(default=1.5, alias="TRACK_MAX_LINK_DISTANCE")

    # --- downscaling ---
    upscale_factor: int = Field(default=2, alias="UPSCALE_FACTOR")
    unet_epochs: int = 60
    diffusion_steps: int = Field(default=50, alias="DIFFUSION_STEPS")
    diffusion_epochs: int = 400
    diffusion_lr: float = 2e-3
    physics_loss_weight: float = 0.15

    # --- severity tiers ---
    severity_low_thresh: float = 0.45
    severity_high_thresh: float = 0.75

    model_config = SettingsConfigDict(env_file=".env", populate_by_name=True, extra="ignore")


settings = Settings()
