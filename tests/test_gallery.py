from io import BytesIO

import pytest

from server.server import app

SAMPLES = [
    ("messi", ["lionel_messi"]),
    ("messi-2", ["lionel_messi"]),
    ("sharapova", ["maria_sharapova"]),
    ("sharapova-2", ["maria_sharapova"]),
    ("federer", ["roger_federer"]),
    ("williams", ["serena_williams"]),
    ("williams-2", ["serena_williams"]),
    ("kohli", ["virat_kohli"]),
    ("kohli-2", ["virat_kohli"]),
    ("tennis-pair", ["maria_sharapova", "serena_williams"]),
]


@pytest.mark.parametrize(("sample", "labels"), SAMPLES)
def test_gallery_sample_runs_through_the_upload_api(sample, labels):
    path = f"/assets/samples/{sample}.webp"
    with app.test_client() as client:
        page = client.get("/")
        assert f'data-sample="{path}"'.encode() in page.data
        with client.get(path) as photo:
            assert photo.status_code == 200
            assert photo.mimetype == "image/webp"
            response = client.post(
                "/api/v1/classify",
                data={"image": (BytesIO(photo.data), f"{sample}.webp")},
            )

    assert response.status_code == 200
    assert response.json["detected_faces"] == len(labels)
    assert sorted(face["label"] for face in response.json["faces"]) == sorted(labels)
