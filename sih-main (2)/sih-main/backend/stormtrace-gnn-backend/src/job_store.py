import json
import threading
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from config import settings

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency in local/dev setups
    redis = None


class BaseJobStore(ABC):
    @abstractmethod
    def create_job(self, job_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def mark_running(self, job_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def set_progress(self, job_id: str, progress: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def mark_done(self, job_id: str, result: Any) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def mark_failed(self, job_id: str, error: str) -> Dict[str, Any]:
        raise NotImplementedError


class MemoryJobStore(BaseJobStore):
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _empty_job() -> Dict[str, Any]:
        return {"status": "queued", "progress": "", "result": None, "error": None}

    def create_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.setdefault(job_id, self._empty_job())
            job["status"] = "queued"
            job["progress"] = ""
            job["result"] = None
            job["error"] = None
            return dict(job)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return dict(job)

    def mark_running(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.setdefault(job_id, self._empty_job())
            job["status"] = "running"
            return dict(job)

    def set_progress(self, job_id: str, progress: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.setdefault(job_id, self._empty_job())
            job["progress"] = progress
            return dict(job)

    def mark_done(self, job_id: str, result: Any) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.setdefault(job_id, self._empty_job())
            job["status"] = "done"
            job["result"] = result
            job["error"] = None
            return dict(job)

    def mark_failed(self, job_id: str, error: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.setdefault(job_id, self._empty_job())
            job["status"] = "failed"
            job["error"] = error
            return dict(job)


class RedisJobStore(BaseJobStore):
    def __init__(self):
        if redis is None:
            raise RuntimeError("redis package is not installed")
        self.client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self._namespace = "stormtrace:jobs"
        try:
            self.client.ping()
        except Exception as exc:  # pragma: no cover - exercised only in runtime envs with no Redis
            raise RuntimeError(f"Redis unavailable at {settings.redis_url}: {exc}") from exc

    def _key(self, job_id: str) -> str:
        return f"{self._namespace}:{job_id}"

    @staticmethod
    def _empty_job() -> Dict[str, Any]:
        return {"status": "queued", "progress": "", "result": None, "error": None}

    def create_job(self, job_id: str) -> Dict[str, Any]:
        job = self._empty_job()
        self.client.set(self._key(job_id), json.dumps(job))
        return dict(job)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        value = self.client.get(self._key(job_id))
        if value is None:
            return None
        return json.loads(value)

    def _update_job(self, job_id: str, **updates: Any) -> Dict[str, Any]:
        job = self.get_job(job_id) or self._empty_job()
        job.update(updates)
        self.client.set(self._key(job_id), json.dumps(job))
        return dict(job)

    def mark_running(self, job_id: str) -> Dict[str, Any]:
        return self._update_job(job_id, status="running")

    def set_progress(self, job_id: str, progress: str) -> Dict[str, Any]:
        return self._update_job(job_id, progress=progress)

    def mark_done(self, job_id: str, result: Any) -> Dict[str, Any]:
        return self._update_job(job_id, status="done", result=result, error=None)

    def mark_failed(self, job_id: str, error: str) -> Dict[str, Any]:
        return self._update_job(job_id, status="failed", error=error)


def get_job_store() -> BaseJobStore:
    backend = (settings.job_store_backend or "memory").lower()
    if backend == "redis":
        try:
            return RedisJobStore()
        except Exception as exc:  # pragma: no cover - runtime fallback path
            warnings.warn(
                f"Redis job store unavailable; falling back to in-memory storage ({exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
            return MemoryJobStore()
    return MemoryJobStore()
