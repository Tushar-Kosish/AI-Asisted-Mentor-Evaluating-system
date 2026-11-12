"""GitHub Reader microservice exposing repository README and engineering signal extraction."""

from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from github_reader.extractor import extract_github_features  # noqa: E402

app = Flask(__name__)


@app.route("/read-repo", methods=["POST"])
def read_repo():
    data = request.get_json(silent=True) or {}
    github_link = data.get("github_link")
    if not github_link:
        return jsonify({"error": "github_link is required"}), 400

    result = extract_github_features(github_link)
    status_code = 200 if result.get("success") else 502
    return jsonify(result), status_code


@app.route("/health")
def health():
    return {"status": "healthy", "service": "github-reader"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
