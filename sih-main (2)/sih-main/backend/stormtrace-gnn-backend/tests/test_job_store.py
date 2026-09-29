from src.job_store import MemoryJobStore


def test_memory_job_store_round_trip():
    store = MemoryJobStore()
    job_id = "job-123"

    store.create_job(job_id)
    assert store.get_job(job_id)["status"] == "queued"

    store.mark_running(job_id)
    store.set_progress(job_id, "running pipeline")
    store.mark_done(job_id, {"status": "ok"})

    job = store.get_job(job_id)
    assert job["status"] == "done"
    assert job["result"]["status"] == "ok"
    assert job["progress"] == "running pipeline"
