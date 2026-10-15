import pytest
from fastapi.testclient import TestClient

from src.data.collector import GestureDatasetBuilder
from src.vision.detector import SyntheticBackend


@pytest.fixture()
def client(model_path, monkeypatch):
    import src.api.main as main
    monkeypatch.setattr(main, "_predictor", None)
    monkeypatch.setattr(main.stream_manager, "_shared_predictor", None)
    return TestClient(main.app)


@pytest.fixture(scope="module")
def traj():
    ds = GestureDatasetBuilder.generate_synthetic_dataset(1, participants=1, seed=3)
    return {l: t for t, l in zip(ds.trajectories, ds.labels)}


def test_health_and_signs(client):
    assert client.get("/health").json()["status"] == "ok"
    body = client.get("/signs").json()
    assert body["count"] == 5 and "tap_forceful" in body["classes"]


def test_predict_metric(client, traj):
    r = client.post("/predict", json={"trajectory": traj["tap_forceful"].tolist(), "fps": 30})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["label"] == "tap_forceful" and j["telemetry"]["peak_speed_m_s"] > 0.8


def test_predict_image_space(client, traj):
    sb = SyntheticBackend(traj["tap_forceful"], 640, 480)
    frames = [sb.detect(None).landmarks.tolist() for _ in range(len(traj["tap_forceful"]))]
    r = client.post("/predict", json={"trajectory": frames, "space": "image", "width": 640, "height": 480})
    assert r.status_code == 200 and r.json()["label"] == "tap_forceful", r.text


@pytest.mark.parametrize("payload", [
    {"trajectory": [[[0, 0, 0]] * 21] * 3},                  # too few frames
    {"trajectory": [[[0, 0, 0]] * 20] * 10},                 # wrong landmark count
    {"trajectory": [[[0, 0]] * 21] * 10},                    # wrong coordinate count
    {"trajectory": [[[0, 0, 0]] * 21] * 10, "fps": 0},       # bad fps
    {},
])
def test_predict_validation(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_websocket_stream(client, traj):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"hello": 1})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"landmarks": [[0, 0, 0]] * 5})
        assert ws.receive_json()["type"] == "error"
        last = None
        for frame in traj["tap_forceful"]:
            ws.send_json({"landmarks": frame.tolist()})
        for _ in range(len(traj["tap_forceful"]) - 14):          # one update per frame after warm-up
            last = ws.receive_json()
        assert last["type"] == "inference_update" and last["prediction"] in last["probabilities"]


def test_cors_headers(client):
    r = client.options("/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
