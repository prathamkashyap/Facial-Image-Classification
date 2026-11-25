"""Shared image decoding, face detection, and model inference."""

import base64
import binascii
import json
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from pickle import UnpicklingError
from threading import local

import cv2
import joblib
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from server.wavelet import w2d

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MAX_IMAGE_SIDE = 6_000
MAX_FACES = 8
IMAGE_FORMATS = ("JPEG", "PNG", "WEBP")
FEATURE_COUNT = 32 * 32 * 4
_thread = local()


class ImageInputError(ValueError):
    def __init__(self, code, message, status):
        super().__init__(message)
        self.code = code
        self.status = status


class ClassifierUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_saved_artifacts():
    directory = Path(os.environ.get("CLASSIFIER_ARTIFACT_DIR", ARTIFACT_DIR))
    try:
        with (directory / "class_dictionary.json").open(encoding="utf-8") as file:
            labels = json.load(file)
        # This operator-controlled directory must contain trusted files.
        model = joblib.load(directory / "saved_model.pkl")
        indices = list(labels.values())
        if (
            not indices
            or any(type(index) is not int for index in indices)
            or len(set(indices)) != len(labels)
            or set(model.classes_) != set(indices)
            or model.n_features_in_ != FEATURE_COUNT
        ):
            raise ValueError("The class dictionary and fitted model are incompatible.")
    except (
        OSError,
        EOFError,
        UnpicklingError,
        AttributeError,
        ImportError,
        TypeError,
        ValueError,
    ) as error:
        raise ClassifierUnavailable(
            "Classifier artifacts could not be loaded."
        ) from error
    return model, {index: label for label, index in labels.items()}


def _cascades():
    if not hasattr(_thread, "cascades"):
        folder = Path(cv2.data.haarcascades)
        face = cv2.CascadeClassifier(
            str(folder / "haarcascade_frontalface_default.xml")
        )
        eyes = cv2.CascadeClassifier(str(folder / "haarcascade_eye.xml"))
        if face.empty() or eyes.empty():
            raise ClassifierUnavailable("OpenCV face or eye cascade is unavailable.")
        # CascadeClassifier uses mutable native state. Each request thread gets its own pair.
        _thread.cascades = (face, eyes)
    return _thread.cascades


def check_ready():
    load_saved_artifacts()
    _cascades()


def decode_image(data):
    if not data:
        raise ImageInputError("missing_image", "Choose an image to classify.", 400)
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageInputError(
            "image_too_large", "Choose an image that is 8 MiB or smaller.", 413
        )

    try:
        with Image.open(BytesIO(data), formats=IMAGE_FORMATS) as source:
            width, height = source.size
            if width * height > MAX_IMAGE_PIXELS or max(width, height) > MAX_IMAGE_SIDE:
                raise Image.DecompressionBombError
            image = ImageOps.exif_transpose(source).convert("RGB")
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except Image.DecompressionBombError as error:
        raise ImageInputError(
            "image_dimensions",
            "Choose an image with at most 12 megapixels and 6,000 pixels per side.",
            413,
        ) from error
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise ImageInputError(
            "invalid_image", "Use a readable JPEG, PNG, or WebP image.", 422
        ) from error


def _decode_base64(value):
    if not value:
        raise ImageInputError("missing_image", "Choose an image to classify.", 400)
    encoded = value.split(",", 1)[-1] if value.startswith("data:") else value
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ImageInputError(
            "image_too_large", "Choose an image that is 8 MiB or smaller.", 413
        )
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ImageInputError(
            "invalid_base64", "The image data is invalid base64.", 400
        ) from error


def read_image_file(path):
    with Path(path).open("rb") as file:
        return decode_image(file.read(MAX_IMAGE_BYTES + 1))


def detect_faces(image):
    """Return qualifying (x, y, width, height) boxes in the decoded image."""
    face_detector, eye_detector = _cascades()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = []
    detected = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for x, y, w, h in detected:
        if len(eye_detector.detectMultiScale(gray[y : y + h, x : x + w])) >= 2:
            boxes.append((int(x), int(y), int(w), int(h)))
    return sorted(boxes, key=lambda box: (-box[2] * box[3], box[1], box[0]))


def face_features(crop):
    raw = cv2.resize(crop, (32, 32))
    wavelet = cv2.resize(w2d(crop, "db1", 5), (32, 32))
    return np.concatenate((raw.ravel(), wavelet.ravel())).astype(np.float64)


def classify_faces(image):
    model, names = load_saved_artifacts()
    height, width = image.shape[:2]
    boxes = detect_faces(image)
    faces = []
    if boxes:
        features = np.stack(
            [
                face_features(image[y : y + h, x : x + w])
                for x, y, w, h in boxes[:MAX_FACES]
            ]
        )
        for box, label, probabilities in zip(
            boxes[:MAX_FACES],
            model.predict(features),
            model.predict_proba(features),
            strict=True,
        ):
            scores = sorted(
                zip(probabilities, model.classes_, strict=True), reverse=True
            )
            faces.append(
                {
                    "box": dict(zip(("x", "y", "width", "height"), box, strict=True)),
                    "label": names[label],
                    "scores": [
                        {
                            "label": names[index],
                            "score": round(float(probability) * 100, 2),
                        }
                        for probability, index in scores
                    ],
                }
            )
    return {
        "image": {"width": width, "height": height},
        "detected_faces": len(boxes),
        "faces": faces,
    }


def classify_bytes(data):
    return classify_faces(decode_image(data))


def classify_image(image_base64_data, file_path=None):
    """Return the legacy result format for a base64 string or a local image path."""
    image = (
        read_image_file(file_path)
        if file_path
        else decode_image(_decode_base64(image_base64_data))
    )
    _, names = load_saved_artifacts()
    labels = [names[index] for index in sorted(names)]
    dictionary = {label: index for index, label in enumerate(labels)}
    results = []
    for face in classify_faces(image)["faces"]:
        scores = {score["label"]: score["score"] for score in face["scores"]}
        results.append(
            {
                "class": face["label"],
                "class_probability": [scores[label] for label in labels],
                "class_dictionary": dictionary,
            }
        )
    return results
