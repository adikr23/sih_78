"""
Persistence layer. Two tables:
  events  -- one row per pipeline run / historical replay case (Amphan, a
             heatwave episode, a live run), so alerts can be grouped and
             event-based holdouts (blueprint section 12, "Scientific leakage"
             guardrail) are natural queries instead of ad-hoc filtering.
  alerts  -- one row per emitted severity-tiered alert, FK'd to its event,
             carrying the full provenance (EFI, physics score, both baseline
             and diffusion peak values) so the amplitude-retention claim is
             independently auditable straight from the DB, not just printed.

Pool sizing is set explicitly (rather than left at SQLAlchemy defaults) since
under concurrent API load this is the most efficiency-relevant DB setting for
a service this shape: a handful of short-lived write-heavy connections.
"""
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Index
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

from config import settings

is_sqlite = settings.database_url.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}
engine_kwargs = {
    "connect_args": connect_args,
    "pool_pre_ping": not is_sqlite,
}
if not is_sqlite:
    engine_kwargs.update({
        "pool_size": 5,
        "max_overflow": 10,
        "pool_recycle": 1800,
    })
engine = create_engine(settings.database_url, **engine_kwargs)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class EventRecord(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    name = Column(String, index=True, unique=False)
    is_historical_replay = Column(Integer, default=0)  # 0/1 boolean, kept simple/portable
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    alerts = relationship("AlertRecord", back_populates="event", cascade="all, delete-orphan")


class AlertRecord(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey("events.id"), index=True)

    centroid_row = Column(Integer)
    centroid_col = Column(Integer)
    peak_value = Column(Float)
    severity = Column(String, index=True)
    confidence = Column(Float)
    radius_km = Column(Float)

    source_provider = Column(String, default="public_weather")
    source_label = Column(String, default="public-weather-forecast")
    source_description = Column(String, default="")

    physics_score = Column(Float)
    efi_peak = Column(Float)
    unet_peak = Column(Float)
    diffusion_peak = Column(Float)
    amplitude_gain_pct = Column(Float)  # (diffusion_peak - unet_peak) / |unet_peak| * 100

    x_min = Column(Float); x_max = Column(Float)
    y_min = Column(Float); y_max = Column(Float)
    z_min = Column(Float); z_max = Column(Float)
    t_start = Column(Integer); t_end = Column(Integer)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    event = relationship("EventRecord", back_populates="alerts")


Index("ix_alerts_severity_created", AlertRecord.severity, AlertRecord.created_at)


def init_db():
    Base.metadata.create_all(engine)
    if is_sqlite:
        with engine.begin() as conn:
            existing = conn.exec_driver_sql("PRAGMA table_info(alerts)").fetchall()
            cols = {row[1] for row in existing}
            if "source_provider" not in cols:
                conn.exec_driver_sql("ALTER TABLE alerts ADD COLUMN source_provider VARCHAR")
            if "source_label" not in cols:
                conn.exec_driver_sql("ALTER TABLE alerts ADD COLUMN source_label VARCHAR")
            if "source_description" not in cols:
                conn.exec_driver_sql("ALTER TABLE alerts ADD COLUMN source_description VARCHAR")


def get_or_create_event(session: Session, name: str, is_historical_replay: bool = False) -> EventRecord:
    event = session.query(EventRecord).filter_by(name=name).order_by(EventRecord.id.desc()).first()
    if event:
        return event
    event = EventRecord(name=name, is_historical_replay=int(is_historical_replay))
    session.add(event)
    session.commit()
    session.refresh(event)
    return event


def save_alert(alert_dict: dict, event_name: str, is_historical_replay: bool = False) -> AlertRecord:
    session = SessionLocal()
    try:
        event = get_or_create_event(session, event_name, is_historical_replay)
        bbox = alert_dict.get("bounding_box") or {}
        record = AlertRecord(
            event_id=event.id,
            centroid_row=alert_dict["centroid_row"],
            centroid_col=alert_dict["centroid_col"],
            peak_value=alert_dict["peak_value"],
            severity=alert_dict["severity"],
            confidence=alert_dict.get("confidence", 0.0),
            radius_km=alert_dict["radius_km"],
            source_provider=alert_dict.get("source_provider") or settings.data_provider,
            source_label=alert_dict.get("source_label") or alert_dict.get("data_source") or "public-weather-forecast",
            source_description=alert_dict.get("source_description") or "",
            physics_score=alert_dict.get("physics_score", 0.0),
            efi_peak=alert_dict.get("efi_peak", 0.0),
            unet_peak=alert_dict.get("unet_peak", 0.0),
            diffusion_peak=alert_dict.get("diffusion_peak", 0.0),
            amplitude_gain_pct=alert_dict.get("amplitude_gain_pct", 0.0),
            x_min=bbox.get("x_min"), x_max=bbox.get("x_max"),
            y_min=bbox.get("y_min"), y_max=bbox.get("y_max"),
            z_min=bbox.get("z_min"), z_max=bbox.get("z_max"),
            t_start=bbox.get("t_start"), t_end=bbox.get("t_end"),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
    finally:
        session.close()


def get_latest_alerts(limit: int = 10, severity: Optional[str] = None, event_name: Optional[str] = None) -> List[dict]:
    session = SessionLocal()
    try:
        q = session.query(AlertRecord).join(EventRecord)
        if severity:
            q = q.filter(AlertRecord.severity == severity)
        if event_name:
            q = q.filter(EventRecord.name == event_name)
        records = q.order_by(AlertRecord.id.desc()).limit(limit).all()
        return [_serialize(r) for r in records]
    finally:
        session.close()


def get_alert(alert_id: int) -> Optional[dict]:
    session = SessionLocal()
    try:
        r = session.query(AlertRecord).filter_by(id=alert_id).first()
        return _serialize(r) if r else None
    finally:
        session.close()


def get_metrics() -> dict:
    session = SessionLocal()
    try:
        total = session.query(AlertRecord).count()
        by_severity = {}
        for sev in ("low", "moderate", "severe"):
            by_severity[sev] = session.query(AlertRecord).filter_by(severity=sev).count()
        avg_physics = session.query(AlertRecord).with_entities(AlertRecord.physics_score).all()
        avg_physics_score = (
            round(sum(v[0] for v in avg_physics) / len(avg_physics), 3) if avg_physics else 0.0
        )
        avg_gain = session.query(AlertRecord).with_entities(AlertRecord.amplitude_gain_pct).all()
        avg_amp_gain = (
            round(sum(v[0] for v in avg_gain) / len(avg_gain), 2) if avg_gain else 0.0
        )
        return {
            "total_alerts": total,
            "by_severity": by_severity,
            "avg_physics_score": avg_physics_score,
            "avg_diffusion_vs_unet_amplitude_gain_pct": avg_amp_gain,
        }
    finally:
        session.close()


def _serialize(r: AlertRecord) -> dict:
    return {
        "id": r.id,
        "event_name": r.event.name if r.event else "",
        "centroid_row": r.centroid_row,
        "centroid_col": r.centroid_col,
        "peak_value": r.peak_value,
        "severity": r.severity,
        "confidence": r.confidence,
        "radius_km": r.radius_km,
        "source_provider": r.source_provider,
        "source_label": r.source_label,
        "source_description": r.source_description,
        "physics_score": r.physics_score,
        "efi_peak": r.efi_peak,
        "unet_peak": r.unet_peak,
        "diffusion_peak": r.diffusion_peak,
        "amplitude_gain_pct": r.amplitude_gain_pct,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }
