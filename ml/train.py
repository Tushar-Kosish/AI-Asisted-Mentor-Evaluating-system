"""Train, compare, evaluate, and export ML models for mentor project evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import FEATURE_DISPLAY_NAMES, NUMERIC_FEATURES, TEXT_COLUMN
from ml.generate_dataset import DATA_PATH, generate_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "ml" / "artifacts"


def _build_preprocessor() -> ColumnTransformer:
    """Create a fresh preprocessing ColumnTransformer for tabular + text features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    text_pipeline = TfidfVectorizer(
        max_features=50,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=5,
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("txt", text_pipeline, TEXT_COLUMN),
        ],
        sparse_threshold=0.0,
    )


def _extract_feature_importances(pipeline: Pipeline, top_n: int = 12) -> list[dict[str, Any]]:
    """Extract readable feature importances or coefficient magnitudes from a fitted pipeline."""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    raw_names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        importances = np.abs(coef.ravel())
    else:
        return []

    total = float(importances.sum()) or 1.0
    items: list[dict[str, Any]] = []
    for raw_name, score in zip(raw_names, importances):
        if raw_name.startswith("num__"):
            key = raw_name.replace("num__", "", 1)
            label = FEATURE_DISPLAY_NAMES.get(key, key)
            kind = "repository"
        else:
            term = raw_name.replace("txt__", "", 1)
            label = f'Text: "{term}"'
            kind = "text"
        items.append(
            {
                "feature": label,
                "raw_name": str(raw_name),
                "kind": kind,
                "importance": round(float(score / total), 4),
            }
        )

    items.sort(key=lambda item: item["importance"], reverse=True)
    return items[:top_n]


def _train_classifiers(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[Pipeline, str, list[dict[str, Any]], list[list[int]]]:
    """Train and compare classification candidates, returning the best fitted pipeline."""
    candidates = {
        "LogisticRegression": LogisticRegression(C=0.4, max_iter=1000, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=160, max_depth=12, min_samples_leaf=2, n_jobs=-1, random_state=42
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=110, learning_rate=0.08, max_depth=3, max_features="sqrt", random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: list[dict[str, Any]] = []
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_cv_score = -1.0
    best_cm: list[list[int]] = []

    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", _build_preprocessor()), ("model", estimator)])
        cv_scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring={"f1": "f1", "roc_auc": "roc_auc", "accuracy": "accuracy"},
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "cv_accuracy": round(float(np.mean(cv_scores["test_accuracy"])), 4),
            "cv_f1": round(float(np.mean(cv_scores["test_f1"])), 4),
            "cv_roc_auc": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
            "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "test_precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "test_recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "test_f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "test_roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        }
        results.append(metrics)

        selection_score = metrics["cv_roc_auc"] + metrics["cv_f1"]
        if selection_score > best_cv_score:
            best_cv_score = selection_score
            best_name = name
            best_pipeline = pipe
            best_cm = confusion_matrix(y_test, preds).tolist()

    assert best_pipeline is not None
    return best_pipeline, best_name, results, best_cm


def _train_regressors(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[Pipeline, str, list[dict[str, Any]], np.ndarray]:
    """Train and compare regression candidates for a 0-10 score target."""
    candidates = {
        "Ridge": Ridge(alpha=2.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=160, max_depth=12, min_samples_leaf=2, n_jobs=-1, random_state=42
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=110, learning_rate=0.07, max_depth=3, max_features="sqrt", random_state=42
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results: list[dict[str, Any]] = []
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_cv_mae = float("inf")
    best_preds = np.zeros(len(y_test))

    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", _build_preprocessor()), ("model", estimator)])
        cv_scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        )
        pipe.fit(X_train, y_train)
        preds = np.clip(pipe.predict(X_test), 0.0, 10.0)

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        cv_mae = float(-np.mean(cv_scores["test_mae"]))

        results.append(
            {
                "model": name,
                "cv_mae": round(cv_mae, 4),
                "cv_r2": round(float(np.mean(cv_scores["test_r2"])), 4),
                "test_mae": round(mae, 4),
                "test_rmse": round(rmse, 4),
                "test_r2": round(r2, 4),
            }
        )

        if cv_mae < best_cv_mae:
            best_cv_mae = cv_mae
            best_name = name
            best_pipeline = pipe
            best_preds = preds

    assert best_pipeline is not None
    return best_pipeline, best_name, results, best_preds


def _save_charts(
    classifier_results: list[dict[str, Any]],
    cm: list[list[int]],
    top_features: list[dict[str, Any]],
    y_inn_test: pd.Series,
    y_inn_pred: np.ndarray,
    y_imp_test: pd.Series,
    y_imp_pred: np.ndarray,
    artifacts_dir: Path,
) -> None:
    """Generate clean evaluation charts for academic reporting and the web dashboard."""
    plt.style.use("dark_background")

    # 1. Classifier comparison
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=140)
    names = [r["model"].replace("Classifier", "") for r in classifier_results]
    x = np.arange(len(names))
    width = 0.25
    ax.bar(x - width, [r["test_accuracy"] for r in classifier_results], width, label="Accuracy", color="#38bdf8")
    ax.bar(x, [r["test_f1"] for r in classifier_results], width, label="F1 Score", color="#818cf8")
    ax.bar(x + width, [r["test_roc_auc"] for r in classifier_results], width, label="ROC-AUC", color="#10b981")
    ax.set_ylim(0.5, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title("Acceptance Classifier Comparison (Held-Out Test Set)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(artifacts_dir / "model_comparison.png")
    plt.close(fig)

    # 2. Confusion matrix
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=140)
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Rejected", "Accepted"])
    ax.set_yticklabels(["Rejected", "Accepted"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Test Set Confusion Matrix")
    for i in range(2):
        for j in range(2):
            val = int(cm_arr[i, j])
            color = "black" if val > cm_arr.max() * 0.55 else "white"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(artifacts_dir / "confusion_matrix.png")
    plt.close(fig)

    # 3. Feature importance
    ordered = list(reversed(top_features[:10]))
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=140)
    labels = [item["feature"] for item in ordered]
    values = [item["importance"] for item in ordered]
    colors = ["#34d399" if item["kind"] == "repository" else "#a78bfa" for item in ordered]
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Relative Importance")
    ax.set_title("Top 10 Features Driving Acceptance Predictions")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(artifacts_dir / "feature_importance.png")
    plt.close(fig)

    # 4. Score predictions scatter
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=140)
    for ax, y_true, y_pred, title, color in [
        (axes[0], y_inn_test, y_inn_pred, "Innovation Score (0-10)", "#38bdf8"),
        (axes[1], y_imp_test, y_imp_pred, "Impact Score (0-10)", "#c084fc"),
    ]:
        ax.scatter(y_true, y_pred, alpha=0.65, color=color, edgecolors="none")
        ax.plot([1, 10], [1, 10], "--", color="#94a3b8", linewidth=1.2, label="Ideal y = x")
        ax.set_xlim(1, 10)
        ax.set_ylim(1, 10)
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(artifacts_dir / "score_predictions.png")
    plt.close(fig)


def train_all(data_path: Path = DATA_PATH, artifacts_dir: Path = ARTIFACTS_DIR) -> dict[str, Any]:
    """Run the full training, model comparison, evaluation, and artifact export pipeline."""
    if not data_path.exists():
        generate_dataset(output_path=data_path)

    df = pd.read_csv(data_path)
    feature_cols = NUMERIC_FEATURES + [TEXT_COLUMN]
    X = df[feature_cols]
    y_accept = df["accepted"].astype(int)
    y_innovation = df["score_innovation"].astype(float)
    y_impact = df["score_impact"].astype(float)

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=y_accept, random_state=42
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    clf_pipe, best_clf_name, clf_results, cm = _train_classifiers(
        X_train, X_test, y_accept.iloc[train_idx], y_accept.iloc[test_idx]
    )
    inn_pipe, best_inn_name, inn_results, inn_preds = _train_regressors(
        X_train, X_test, y_innovation.iloc[train_idx], y_innovation.iloc[test_idx]
    )
    imp_pipe, best_imp_name, imp_results, imp_preds = _train_regressors(
        X_train, X_test, y_impact.iloc[train_idx], y_impact.iloc[test_idx]
    )

    top_features = _extract_feature_importances(clf_pipe, top_n=12)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf_pipe, artifacts_dir / "acceptance_model.joblib")
    joblib.dump(inn_pipe, artifacts_dir / "innovation_model.joblib")
    joblib.dump(imp_pipe, artifacts_dir / "impact_model.joblib")

    _save_charts(
        clf_results,
        cm,
        top_features,
        y_innovation.iloc[test_idx],
        inn_preds,
        y_impact.iloc[test_idx],
        imp_preds,
        artifacts_dir,
    )

    metrics_payload: dict[str, Any] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(data_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "total_rows": int(len(df)),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "acceptance_rate": round(float(y_accept.mean()), 4),
            "numeric_features": len(NUMERIC_FEATURES),
            "text_features": "TF-IDF (up to 50 1-2 ngrams)",
        },
        "classification": {
            "selected_model": best_clf_name,
            "candidates": clf_results,
            "confusion_matrix": cm,
            "labels": ["REJECTED", "ACCEPTED"],
        },
        "regression_innovation": {
            "selected_model": best_inn_name,
            "candidates": inn_results,
        },
        "regression_impact": {
            "selected_model": best_imp_name,
            "candidates": imp_results,
        },
        "top_features": top_features,
        "charts": [
            "model_comparison.png",
            "confusion_matrix.png",
            "feature_importance.png",
            "score_predictions.png",
        ],
    }

    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics_payload


if __name__ == "__main__":
    summary = train_all()
    best_clf = next(
        c for c in summary["classification"]["candidates"]
        if c["model"] == summary["classification"]["selected_model"]
    )
    best_inn = next(
        c for c in summary["regression_innovation"]["candidates"]
        if c["model"] == summary["regression_innovation"]["selected_model"]
    )
    best_imp = next(
        c for c in summary["regression_impact"]["candidates"]
        if c["model"] == summary["regression_impact"]["selected_model"]
    )
    print(f"Best Classifier: {best_clf['model']} | Accuracy={best_clf['test_accuracy']:.3f} F1={best_clf['test_f1']:.3f} ROC-AUC={best_clf['test_roc_auc']:.3f}")
    print(f"Best Innovation Regressor: {best_inn['model']} | MAE={best_inn['test_mae']:.3f} R2={best_inn['test_r2']:.3f}")
    print(f"Best Impact Regressor: {best_imp['model']} | MAE={best_imp['test_mae']:.3f} R2={best_imp['test_r2']:.3f}")
    print(f"Saved models, metrics.json, and 4 PNG charts to {ARTIFACTS_DIR}")
