"""Flask entry point for the browser demo and its inference API."""

import os
from pathlib import Path

from flask import Flask, jsonify, redirect, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from server import util

UI_DIRECTORY = Path(__file__).resolve().parents[1] / "UI"


def create_app():
    app = Flask(__name__, static_folder=str(UI_DIRECTORY), static_url_path="/assets")
    app.config.update(
        MAX_CONTENT_LENGTH=12 * 1024 * 1024,
        MAX_FORM_MEMORY_SIZE=12 * 1024 * 1024,
    )

    @app.get("/")
    def index():
        return redirect(url_for("static", filename="app.html"))

    @app.get("/healthz")
    def health():
        try:
            util.check_ready()
        except util.ClassifierUnavailable:
            app.logger.exception("Classifier readiness check failed.")
            return jsonify(status="unavailable"), 503
        return jsonify(status="ok")

    @app.post("/api/v1/classify")
    def classify():
        if request.mimetype != "multipart/form-data":
            return _error(
                "unsupported_request", "Send a multipart form with an image field.", 415
            )
        images = request.files.getlist("image")
        if not images:
            return _error("missing_image", "Choose an image to classify.", 400)
        if len(images) != 1:
            return _error("multiple_images", "Send one image per request.", 400)
        return jsonify(
            util.classify_bytes(images[0].stream.read(util.MAX_IMAGE_BYTES + 1))
        )

    @app.post("/classify_image")
    def classify_legacy():
        response = jsonify(util.classify_image(request.form.get("image_data")))
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    @app.errorhandler(util.ImageInputError)
    def image_error(error):
        return _error(error.code, str(error), error.status)

    @app.errorhandler(util.ClassifierUnavailable)
    def model_unavailable(error):
        app.logger.error("Classifier unavailable.", exc_info=error)
        return _error(
            "classifier_unavailable", "The classifier is temporarily unavailable.", 503
        )

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_error_detail):
        return _error(
            "request_too_large", "The upload exceeds the request size limit.", 413
        )

    return app


def _error(code, message, status):
    return jsonify(error={"code": code, "message": message}), status


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
