"""API tests through the HTTP layer using FastAPI's test client."""

from fastapi.testclient import TestClient

from vramcalc.main import app

client = TestClient(app)


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_models():
    resp = client.get("/models")
    assert resp.status_code == 200
    names = [m["name"] for m in resp.json()]
    assert "llama-3.1-8b" in names


def test_estimate_with_preset():
    resp = client.post("/estimate", json={"preset": "llama-3.1-8b"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "llama-3.1-8b"
    assert body["total_gib"] > body["weights_gib"]


def test_estimate_with_custom_spec():
    resp = client.post(
        "/estimate",
        json={
            "spec": {
                "name": "my-model",
                "params_billion": 1.0,
                "num_layers": 16,
                "num_kv_heads": 4,
                "head_dim": 64,
            },
            "weight_dtype": "int4",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "my-model"


def test_unknown_preset_is_404():
    resp = client.post("/estimate", json={"preset": "gpt-9000"})
    assert resp.status_code == 404


def test_missing_preset_and_spec_is_422():
    resp = client.post("/estimate", json={})
    assert resp.status_code == 422


def test_negative_context_rejected_by_validation():
    resp = client.post(
        "/estimate", json={"preset": "llama-3.1-8b", "context_length": -1}
    )
    assert resp.status_code == 422
