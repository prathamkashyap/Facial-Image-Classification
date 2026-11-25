import json

import pytest
from PIL import Image

from server import train, util


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / "train"
    labels = json.loads(
        (util.ARTIFACT_DIR / "class_dictionary.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(util, "detect_faces", lambda _image: [(0, 0, 48, 48)])
    for index, name in enumerate(labels):
        folder = root / name
        folder.mkdir(parents=True)
        for example in range(8):
            color = (index * 40, example * 20, 10)
            Image.new("RGB", (48, 48), color).save(folder / f"{example}.png")
    return root, labels


def test_training_deduplicates_and_writes_a_loadable_bundle(
    dataset, tmp_path, monkeypatch
):
    root, labels = dataset
    first, second = list(labels)[:2]
    original = (root / first / "0.png").read_bytes()
    (root / first / "duplicate.png").write_bytes(original)
    (root / second / "conflict.png").write_bytes(original)
    (root / second / "bad.png").write_bytes(b"invalid")
    output = tmp_path / "run"

    report = train.train(root, output)
    assert report["skipped"] == {
        "conflicting_labels": 2,
        "duplicate_features": 1,
        "unreadable_or_out_of_limits": 1,
    }
    assert report["training_samples"] + report["test_samples"] == 39
    assert report["class_order"] == list(labels)
    assert sum(map(sum, report["confusion_matrix"])) == report["test_samples"]
    assert (
        json.loads((output / "evaluation.json").read_text(encoding="utf-8")) == report
    )

    util.load_saved_artifacts.cache_clear()
    monkeypatch.setenv("CLASSIFIER_ARTIFACT_DIR", str(output))
    try:
        result = util.classify_bytes((root / first / "2.png").read_bytes())
        assert result["detected_faces"] == 1
        assert {score["label"] for score in result["faces"][0]["scores"]} == set(labels)
    finally:
        util.load_saved_artifacts.cache_clear()

    with pytest.raises(FileExistsError, match="already exists"):
        train.train(root, output)


def test_insufficient_class_data_produces_no_artifacts(dataset, tmp_path):
    root, labels = dataset
    folder = root / next(iter(labels))
    for path in folder.glob("*.png"):
        path.unlink()
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="usable, distinct photos per class"):
        train.train(root, output)
    assert not output.exists()
