"""Video Parser microservice exposing pitch video metadata extraction."""

from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_parser.extractor import extract_video_summary  # noqa: E402

app = Flask(__name__)


@app.route("/parse-video", methods=["POST"])
def parse_video():
    data = request.get_json(silent=True) or {}
    video_url = data.get("video_link")
    if not video_url:
        return jsonify({"error": "video_link is required"}), 400

    result = extract_video_summary(video_url)
    status_code = 200 if result.get("success") else 502
    return jsonify(result), status_code


@app.route("/health")
def health():
    return {"status": "healthy", "service": "video-parser"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)
