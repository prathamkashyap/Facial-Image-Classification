import base64
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from server import util
from server.server import app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def post_image(client, data):
    return client.post("/api/v1/classify", data={"image": (BytesIO(data), "photo.jpg")})


def image_bytes(width=120, height=90, orientation=None):
    stream = BytesIO()
    image = Image.new("RGB", (width, height), "white")
    exif = image.getexif()
    if orientation is not None:
        exif[274] = orientation
    image.save(stream, format="JPEG", exif=exif)
    return stream.getvalue()


def test_readiness_and_ui_from_outside_repo(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert client.get("/healthz").json == {"status": "ok"}
    page = client.get("/", follow_redirects=True)
    assert page.status_code == 200
    assert b"Sports Person Classifier" in page.data


def test_multiple_faces_have_valid_boxes_and_ranked_scores(client):
    example = ROOT / "UI/test_images/sharapova_3.jpg"
    response = post_image(client, example.read_bytes())
    assert response.status_code == 200
    result = response.json
    assert result["detected_faces"] == len(result["faces"]) == 2
    assert {face["label"] for face in result["faces"]} == {
        "maria_sharapova",
        "roger_federer",
    }
    for face in result["faces"]:
        box = face["box"]
        assert 0 <= box["x"] < box["x"] + box["width"] <= result["image"]["width"]
        assert 0 <= box["y"] < box["y"] + box["height"] <= result["image"]["height"]
        scores = [entry["score"] for entry in face["scores"]]
        assert len(scores) == 5
        assert scores == sorted(scores, reverse=True)
        assert sum(scores) == pytest.approx(100, abs=0.03)


def test_no_face_uses_exif_oriented_dimensions(client):
    response = post_image(client, image_bytes(120, 90, orientation=6))
    assert response.status_code == 200
    assert response.json == {
        "image": {"width": 90, "height": 120},
        "detected_faces": 0,
        "faces": [],
    }


def test_empty_corrupt_and_non_multipart_requests(client):
    assert post_image(client, b"").json["error"]["code"] == "missing_image"
    corrupt = post_image(client, b"not an image")
    assert (corrupt.status_code, corrupt.json["error"]["code"]) == (
        422,
        "invalid_image",
    )
    assert client.post("/api/v1/classify", json={"image": "text"}).status_code == 415
    assert client.get("/api/v1/classify").status_code == 405


def test_byte_and_pixel_limits(client):
    upload = post_image(client, b"x" * (util.MAX_IMAGE_BYTES + 1))
    assert (upload.status_code, upload.json["error"]["code"]) == (
        413,
        "image_too_large",
    )
    pixels = post_image(client, image_bytes(4000, 3100))
    assert (pixels.status_code, pixels.json["error"]["code"]) == (
        413,
        "image_dimensions",
    )
    request = post_image(client, b"x" * app.config["MAX_CONTENT_LENGTH"])
    assert (request.status_code, request.json["error"]["code"]) == (
        413,
        "request_too_large",
    )


def test_legacy_base64_preserves_score_indices(client):
    data = (ROOT / "server/test_images/federer2.jpg").read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    legacy = client.post(
        "/classify_image", data={"image_data": "data:image/jpeg;base64," + encoded}
    )
    assert legacy.status_code == 200
    assert legacy.headers["Access-Control-Allow-Origin"] == "*"
    assert client.post("/classify_image", data={"image_data": "%%%"}).status_code == 400
    modern = post_image(client, data).json["faces"][0]
    result = legacy.json[0]
    assert result["class"] == modern["label"] == "roger_federer"
    for score in modern["scores"]:
        index = result["class_dictionary"][score["label"]]
        assert result["class_probability"][index] == score["score"]


def test_unavailable_model_is_reported_without_exposing_local_paths(
    client, tmp_path, monkeypatch
):
    monkeypatch.setenv("CLASSIFIER_ARTIFACT_DIR", str(tmp_path))
    util.load_saved_artifacts.cache_clear()
    assert client.get("/healthz").status_code == 503
    failed = post_image(client, image_bytes())
    assert (failed.status_code, failed.json["error"]["code"]) == (
        503,
        "classifier_unavailable",
    )
    assert str(tmp_path) not in failed.get_data(as_text=True)
