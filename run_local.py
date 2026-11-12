"""One-command launcher for the AI-Assisted Mentor Evaluation System."""

from __future__ import annotations

import os
from pathlib import Path

from agent.app import create_app
from ml.evaluator import ARTIFACTS_DIR


def main() -> None:
    required = [
        ARTIFACTS_DIR / "acceptance_model.joblib",
        ARTIFACTS_DIR / "innovation_model.joblib",
        ARTIFACTS_DIR / "impact_model.joblib",
        ARTIFACTS_DIR / "metrics.json",
    ]
    if not all(path.exists() for path in required):
        print("Training ML models on project dataset (first-time setup)...")
        from ml.train import train_all

        train_all(artifacts_dir=ARTIFACTS_DIR)

    port = int(os.getenv("PORT", "5000"))
    app = create_app()
    print(f"\nAI-Assisted Mentor Evaluation System running at: http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
