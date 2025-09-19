import json
import pytest

import server
from server import app as flask_app


@pytest.fixture(scope="module")
def app():
    flask_app.config.update(TESTING=True)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def isolate_saved_path(tmp_path, monkeypatch):
    # Avoid polluting real saved_annotations.jsonl during tests
    test_path = tmp_path / "saved_annotations.jsonl"
    monkeypatch.setattr(server, "SAVED_PATH", str(test_path))
    yield


def get_one_task_id(client, split="training"):
    r = client.get(f"/api/tasks?split={split}")
    assert r.status_code == 200
    data = r.get_json()
    ids = data.get("task_ids") or []
    assert isinstance(ids, list)
    assert len(ids) > 0
    return ids[0]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data.get("ok") is True
    assert "ts" in data


def test_tasks_training_nonempty(client):
    r = client.get("/api/tasks?split=training")
    assert r.status_code == 200
    data = r.get_json()
    assert data.get("split") == "training"
    assert isinstance(data.get("task_ids"), list)
    assert len(data.get("task_ids")) > 0


def test_tasks_evaluation_structure(client):
    r = client.get("/api/tasks?split=evaluation")
    assert r.status_code == 200
    data = r.get_json()
    assert data.get("split") == "evaluation"
    assert isinstance(data.get("task_ids"), list)


def test_get_task_valid_training(client):
    task_id = get_one_task_id(client, split="training")
    r = client.get(f"/api/task/{task_id}?split=training")
    assert r.status_code == 200
    data = r.get_json()
    assert data.get("task_id") == task_id
    assert data.get("split") == "training"
    assert isinstance(data.get("train"), list)
    assert isinstance(data.get("test"), list)
    # For training split, gt may exist
    assert "gt" in data


def test_get_task_not_found(client):
    r = client.get("/api/task/__does_not_exist__?split=training")
    assert r.status_code == 404
    data = r.get_json()
    assert data.get("error") == "task_not_found"


def test_save_invalid_request(client):
    # Missing required fields
    r = client.post("/api/save", json={})
    assert r.status_code == 400
    data = r.get_json()
    assert data.get("error") == "invalid_request"


def test_save_and_list_saved(client):
    # Save a simple (likely incorrect) output for a real training task
    task_id = get_one_task_id(client, split="training")
    task_res = client.get(f"/api/task/{task_id}?split=training")
    assert task_res.status_code == 200
    task_data = task_res.get_json()
    tests = task_data.get("test") or []
    assert len(tests) > 0
    # Use test_index 0; output = input (just for testing persistence)
    test_index = 0
    output_grid = tests[test_index]["input"]

    payload = {
        "task_id": task_id,
        "split": "training",
        "test_index": test_index,
        "output": output_grid,
        "insts": ["example instruction"],
    }
    save_res = client.post("/api/save", json=payload)
    assert save_res.status_code == 200
    save_data = save_res.get_json()
    assert save_data.get("saved") is True
    rec = save_data.get("record") or {}
    assert rec.get("task_id") == task_id
    assert isinstance(rec.get("output_hash"), str)
    # ok may be True/False/None depending on availability of GT and correctness
    assert "ok" in rec

    # Recent saved
    recent_res = client.get("/api/saved?limit=5")
    assert recent_res.status_code == 200
    recent = recent_res.get_json()
    assert isinstance(recent.get("items"), list)
    assert len(recent.get("items")) >= 1


def test_task_hashes_endpoint(client):
    task_id = get_one_task_id(client, split="training")
    r = client.get(f"/api/task_hashes/{task_id}?split=training")
    assert r.status_code == 200
    data = r.get_json()
    assert data.get("task_id") == task_id
    assert "gt_hashes" in data
    assert "saved_output_hashes" in data
    counts = data.get("counts") or {}
    assert "gt" in counts and "saved" in counts


def test_download_saved(client):
    # Should return 200 and serve a file (even if empty)
    r = client.get("/api/download/saved")
    assert r.status_code == 200
    # Content-Type may vary; ensure some bytes are returned
    assert r.data is not None
