import time

import pytest
from fastapi.testclient import TestClient

from podsum import app as app_module


@pytest.fixture(autouse=True)
def isolated_runs_dir(tmp_path, monkeypatch):
    """create_job() makes a real directory before any mocking takes effect (so a job survives a server
    restart). Without this, every test run would leave real folders under the project's own data/runs/web/."""
    monkeypatch.setattr(app_module, "RUNS", tmp_path / "runs" / "web")


def test_url_job_runs_and_returns_result(monkeypatch):
    def fake_run(source, out_dir, method="semantic", on_progress=None, **kw):
        on_progress(1, "Ingest")
        return {"source": "t", "method": method, "tldr": "x",
                "chapters": [{"start": 0.0, "end": 5.0, "title": "A", "summary": "s"}]}

    monkeypatch.setattr(app_module, "run", fake_run)
    client = TestClient(app_module.app)
    job_id = client.post("/api/jobs", data={"url": "https://youtu.be/8jPQjjsBbIc"}).json()["id"]
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "done" and job["media"] == {"type": "youtube", "id": "8jPQjjsBbIc"}
    assert job["result"]["chapters"][0]["title"] == "A"


def test_rejects_empty_and_bad_method():
    client = TestClient(app_module.app)
    assert client.post("/api/jobs", data={}).status_code == 400
    assert client.post("/api/jobs", data={"url": "https://x.com", "method": "nope"}).status_code == 400