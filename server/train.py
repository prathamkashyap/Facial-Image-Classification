"""Train and evaluate a classifier using local, labeled athlete photos."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from server import util

EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".webp"}
MIN_PER_CLASS = 5
TEST_SIZE = 0.25


def collect_examples(dataset, labels):
    by_fingerprint = defaultdict(dict)
    skipped = Counter()
    counts = Counter()

    for name, label in labels.items():
        folder = dataset / name
        if not folder.is_dir():
            raise ValueError(f"Required class directory is missing: {folder}")

        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            counts[name] += 1
            try:
                image = util.read_image_file(path)
            except (OSError, util.ImageInputError):
                skipped["unreadable_or_out_of_limits"] += 1
                continue

            boxes = util.detect_faces(image)
            if len(boxes) != 1:
                skipped[
                    "no_qualifying_face" if not boxes else "multiple_qualifying_faces"
                ] += 1
                continue

            x, y, width, height = boxes[0]
            feature = util.face_features(image[y : y + height, x : x + width])
            fingerprint = hashlib.sha256(feature.tobytes()).digest()
            candidates = by_fingerprint[fingerprint]
            if label in candidates:
                skipped["duplicate_features"] += 1
            else:
                candidates[label] = feature

    features, targets = [], []
    for candidates in by_fingerprint.values():
        if len(candidates) > 1:
            skipped["conflicting_labels"] += len(candidates)
            continue
        ((label, feature),) = candidates.items()
        features.append(feature)
        targets.append(label)

    accepted = Counter(targets)
    short = [
        f"{name} ({accepted[label]})"
        for name, label in labels.items()
        if accepted[label] < MIN_PER_CLASS
    ]
    if short:
        raise ValueError(
            f"At least {MIN_PER_CLASS} usable, distinct photos per class are required. "
            f"Too few in {', '.join(short)}."
        )

    return (
        np.stack(features),
        targets,
        {
            "candidates_by_class": {name: counts[name] for name in labels},
            "accepted_by_class": {
                name: accepted[label] for name, label in labels.items()
            },
            "skipped": dict(sorted(skipped.items())),
        },
    )


def train(dataset, output, seed=42):
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    labels = json.loads(
        (util.ARTIFACT_DIR / "class_dictionary.json").read_text(encoding="utf-8")
    )
    features, targets, report = collect_examples(dataset, labels)
    train_x, test_x, train_y, test_y = train_test_split(
        features, targets, stratify=targets, test_size=TEST_SIZE, random_state=seed
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="newton-cg", max_iter=250, random_state=seed),
    )
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    indices, names = list(labels.values()), list(labels)
    report.update(
        {
            "seed": seed,
            "test_size_requested": TEST_SIZE,
            "training_samples": len(train_y),
            "test_samples": len(test_y),
            "class_order": names,
            "metrics": classification_report(
                test_y,
                predictions,
                labels=indices,
                target_names=names,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                test_y, predictions, labels=indices
            ).tolist(),
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".training-", dir=output.parent) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()
        joblib.dump(model, stage / "saved_model.pkl")
        (stage / "class_dictionary.json").write_text(
            json.dumps(labels, indent=2) + "\n", "utf-8"
        )
        (stage / "evaluation.json").write_text(
            json.dumps(report, indent=2) + "\n", "utf-8"
        )
        stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Root of the five class folders"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="A new directory for the run"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the stratified split"
    )
    args = parser.parse_args()
    try:
        train(args.dataset, args.output, seed=args.seed)
    except (OSError, ValueError, util.ClassifierUnavailable) as error:
        parser.error(str(error))
    print(f"Wrote model, class dictionary, and evaluation to {args.output}")


if __name__ == "__main__":
    main()
