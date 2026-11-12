"""Runtime ML evaluator that loads trained scikit-learn pipelines and explains predictions."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.features import NUMERIC_FEATURES, TEXT_COLUMN, build_feature_row

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "ml" / "artifacts"


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load trained joblib pipelines and metrics, training on first run if missing."""
    acc_path = ARTIFACTS_DIR / "acceptance_model.joblib"
    inn_path = ARTIFACTS_DIR / "innovation_model.joblib"
    imp_path = ARTIFACTS_DIR / "impact_model.joblib"
    metrics_path = ARTIFACTS_DIR / "metrics.json"

    if not (acc_path.exists() and inn_path.exists() and imp_path.exists() and metrics_path.exists()):
        from ml.train import train_all

        train_all(artifacts_dir=ARTIFACTS_DIR)

    return (
        joblib.load(acc_path),
        joblib.load(inn_path),
        joblib.load(imp_path),
        json.loads(metrics_path.read_text(encoding="utf-8")),
    )


def get_ml_metrics() -> dict[str, Any]:
    """Return the latest training & held-out evaluation metrics report."""
    _, _, _, metrics = _load_artifacts()
    return metrics


def _build_diagnostics(
    row: dict[str, Any],
    accept_prob: float,
    score_innovation: float,
    score_impact: float,
) -> tuple[list[str], list[str], list[str]]:
    """Derive grounded strengths, gaps, and targeted mentor questions from extracted signals."""
    positives: list[str] = []
    negatives: list[str] = []
    questions: list[str] = []

    if row["commit_count"] >= 12:
        positives.append(
            f"Consistent engineering history detected ({row['commit_count']} commits across {row['days_active']} active days)."
        )
    elif row["commit_count"] <= 4:
        negatives.append(
            f"Very sparse git commit history ({row['commit_count']} commits), suggesting limited iteration or a late single-upload."
        )
        questions.append(
            "Walk us through your commit timeline—which components were built live versus pre-existing starter code?"
        )

    if row["file_count"] >= 14:
        positives.append(f"Modular repository structure ({row['file_count']} files) indicating non-trivial implementation.")
    elif row["file_count"] <= 6:
        negatives.append(f"Minimal codebase footprint ({row['file_count']} files) with limited modular separation.")

    if row["has_tests"]:
        positives.append("Automated test suite present in repository, strengthening reliability and verification.")
    else:
        negatives.append("No automated tests folder/files detected in the repository.")
        questions.append(
            "How did you validate accuracy and edge cases without an automated test suite in the repository?"
        )

    if row["readme_word_count"] >= 180 and row["has_architecture_docs"]:
        positives.append(
            f"Comprehensive documentation ({row['readme_word_count']} README words) with architecture/setup coverage."
        )
    elif row["readme_word_count"] < 90:
        negatives.append(
            f"Short project documentation ({row['readme_word_count']} words) lacking detailed setup or methodology."
        )
        questions.append(
            "Can you explain your end-to-end architecture and local setup steps that are missing from the README?"
        )

    if row["has_requirements"]:
        positives.append("Reproducible dependency manifest included (e.g., requirements.txt or package.json).")
    else:
        negatives.append("No dependency manifest detected; reproducibility may fail on another machine.")

    if row["has_demo_video"]:
        positives.append("Pitch/demo video provided for walkthrough verification.")
    else:
        negatives.append("No working video demo link detected or metadata could not be verified.")

    if not positives:
        positives.append("Submission matches baseline metadata formatting and problem framing.")
    if not negatives:
        negatives.append("Requires live mentor Q&A to verify originality and real-world deployment readiness.")

    if len(questions) < 3:
        questions.extend(
            [
                f"What measurable evaluation metric (e.g., latency, F1, user time saved) proves your {score_impact:.1f}/10 impact claim?",
                "What happens when an unexpected or low-quality input hits your pipeline in production?",
                "Which component was technically hardest to build, and how would you scale it beyond a prototype?",
            ]
        )

    return positives[:4], negatives[:4], questions[:3]


def _estimate_boilerplate_risk(row: dict[str, Any]) -> float:
    """Estimate template/AI-boilerplate risk (0-100%) from repo maturity vs verbosity."""
    risk = 45.0
    if row["commit_count"] >= 15:
        risk -= 14.0
    elif row["commit_count"] <= 2:
        risk += 18.0
    if row["has_tests"]:
        risk -= 10.0
    if row["file_count"] >= 15:
        risk -= 8.0
    if row["readme_word_count"] > 220 and row["file_count"] <= 5:
        risk += 16.0
    if row["contributors_count"] >= 2:
        risk -= 5.0
    return round(float(np.clip(risk, 8.0, 92.0)), 1)


def evaluate_submission_ml(payload: dict[str, Any]) -> dict[str, Any]:
    """Run trained classification & regression models on normalized submission features."""
    clf_model, inn_model, imp_model, metrics = _load_artifacts()
    row = build_feature_row(payload)
    df_row = pd.DataFrame([row], columns=NUMERIC_FEATURES + [TEXT_COLUMN])

    accept_prob = float(clf_model.predict_proba(df_row)[0][1])
    decision = "ACCEPTED" if accept_prob >= 0.5 else "REJECTED"
    score_innovation = round(float(np.clip(inn_model.predict(df_row)[0], 0.0, 10.0)), 1)
    score_impact = round(float(np.clip(imp_model.predict(df_row)[0], 0.0, 10.0)), 1)

    positives, negatives, questions = _build_diagnostics(
        row, accept_prob, score_innovation, score_impact
    )

    clf_name = metrics["classification"]["selected_model"]
    inn_name = metrics["regression_innovation"]["selected_model"]
    imp_name = metrics["regression_impact"]["selected_model"]

    market_notes = str(payload.get("market_search_results") or "").strip()
    if market_notes:
        necessity_summary = (
            f"ML Impact Estimate: {score_impact:.1f}/10 ({imp_name}). "
            f"Market scan context: {market_notes[:280]}"
        )
    else:
        necessity_summary = (
            f"ML Impact Estimate: {score_impact:.1f}/10 ({imp_name}) based on project scope, "
            f"documentation clarity ({row['readme_word_count']} words), and engineering completeness."
        )

    justification = (
        f"Evaluated by trained ML pipeline ({clf_name} / {inn_name}): "
        f"Acceptance confidence is {accept_prob * 100:.1f}% ({decision}), "
        f"with predicted Innovation {score_innovation:.1f}/10 and Impact {score_impact:.1f}/10 "
        f"across {row['commit_count']} commits and {row['file_count']} repository files."
    )

    feature_signals = {k: row[k] for k in NUMERIC_FEATURES}

    return {
        "decision": decision,
        "acceptance_probability": round(accept_prob * 100.0, 1),
        "score_innovation": score_innovation,
        "score_impact": score_impact,
        "positive_points": positives,
        "negative_points": negatives,
        "real_world_necessity": necessity_summary,
        "ai_generated_percentage": _estimate_boilerplate_risk(row),
        "mentor_questions": questions,
        "justification": justification,
        "feature_signals": feature_signals,
        "evaluation_mode": f"Trained ML Pipeline ({clf_name})",
    }
