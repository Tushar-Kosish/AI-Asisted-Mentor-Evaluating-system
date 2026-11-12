"""Generate a reproducible benchmark dataset of student & hackathon project submissions."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "project_submissions.csv"

_DOMAINS = [
    ("HealthTech", "clinical triage symptom screening patient monitoring medical imaging"),
    ("EdTech", "adaptive learning student assessment mentor feedback quiz evaluation"),
    ("FinTech", "fraud detection credit scoring expense tracking anomaly alert"),
    ("AgriTech", "crop disease detection soil moisture irrigation yield prediction"),
    ("SmartCity", "traffic flow optimization pothole detection waste routing safety"),
    ("ClimateTech", "solar energy forecasting carbon footprint grid load prediction"),
    ("DevTools", "code review automation static analysis test coverage pull request"),
    ("Accessibility", "sign language translation screen reader voice navigation captioning"),
]

_STRONG_TEMPLATES = [
    "{name}: End-to-end {domain} platform using {tech}. Architecture includes data ingestion pipeline, evaluation metrics, REST API, automated tests, and deployment setup instructions for real-world pilot users.",
    "{name}: Built for {domain} workflows with {tech}. Documented methodology, baseline comparisons, cross-validation accuracy benchmarks, interactive dashboard, and clear societal impact.",
    "{name}: Scalable {domain} assistant powered by {tech}. Features modular backend services, reproducible training pipeline, latency benchmarks, setup guide, and live demo walkthrough.",
]

_MEDIUM_TEMPLATES = [
    "{name}: Prototype for {domain} focused on {tech}. Basic working UI and initial model script included; needs more unit tests and clearer installation steps.",
    "{name}: Early MVP exploring {domain} via {tech}. Core notebook and API endpoint implemented, though evaluation metrics and error handling remain limited.",
    "{name}: Student project targeting {domain} using {tech}. Functional demo script provided, with partial documentation and single-branch history.",
]

_WEAK_TEMPLATES = [
    "{name}: Simple starter template for {domain}. Work in progress repository with minimal code files and placeholder readme.",
    "{name}: Initial commit upload. Basic hello world demo script without architecture notes, tests, or dependency setup.",
    "{name}: Hackathon idea draft on {domain}. Mostly slides and boilerplate files; implementation incomplete.",
]

_TECH_STACKS = [
    "scikit-learn RandomForest TF-IDF Flask SQLite",
    "PyTorch CNN transfer learning FastAPI Docker",
    "XGBoost feature engineering Streamlit Plotly",
    "NLP transformer embeddings cosine similarity PostgreSQL",
    "time-series forecasting GradientBoosting Python API",
    "computer vision OpenCV pose estimation React",
]


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(n_samples: int = 650, seed: int = 42, output_path: Path = DATA_PATH) -> pd.DataFrame:
    """Create a realistic labeled dataset of project submissions for model training."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    archetypes = rng.choice(
        ["strong", "promising_sparse_docs", "high_pitch_low_code", "weak"],
        size=n_samples,
        p=[0.36, 0.22, 0.16, 0.26],
    )

    for idx, archetype in enumerate(archetypes, start=1):
        domain_name, domain_terms = _DOMAINS[idx % len(_DOMAINS)]
        tech = _TECH_STACKS[(idx * 3 + int(rng.integers(0, 5))) % len(_TECH_STACKS)]
        project_name = f"{domain_name} MentorGuard #{idx:03d}"

        if archetype == "strong":
            readme_word_count = int(rng.integers(240, 950))
            file_count = int(rng.integers(14, 75))
            commit_count = int(rng.integers(14, 90))
            contributors_count = int(rng.integers(1, 5))
            stars_count = int(rng.integers(0, 28))
            has_tests = int(rng.random() < 0.78)
            has_requirements = int(rng.random() < 0.94)
            has_demo_video = int(rng.random() < 0.88)
            has_ppt = int(rng.random() < 0.76)
            has_architecture_docs = int(rng.random() < 0.90)
            days_active = int(rng.integers(3, 45))
            template = _STRONG_TEMPLATES[int(rng.integers(0, len(_STRONG_TEMPLATES)))]
            quality_bonus = 1.6
        elif archetype == "promising_sparse_docs":
            readme_word_count = int(rng.integers(45, 210))
            file_count = int(rng.integers(10, 42))
            commit_count = int(rng.integers(8, 38))
            contributors_count = int(rng.integers(1, 4))
            stars_count = int(rng.integers(0, 10))
            has_tests = int(rng.random() < 0.42)
            has_requirements = int(rng.random() < 0.75)
            has_demo_video = int(rng.random() < 0.62)
            has_ppt = int(rng.random() < 0.45)
            has_architecture_docs = int(rng.random() < 0.38)
            days_active = int(rng.integers(2, 20))
            template = _MEDIUM_TEMPLATES[int(rng.integers(0, len(_MEDIUM_TEMPLATES)))]
            quality_bonus = 0.25
        elif archetype == "high_pitch_low_code":
            readme_word_count = int(rng.integers(180, 520))
            file_count = int(rng.integers(2, 9))
            commit_count = int(rng.integers(1, 6))
            contributors_count = int(rng.integers(1, 3))
            stars_count = int(rng.integers(0, 3))
            has_tests = int(rng.random() < 0.08)
            has_requirements = int(rng.random() < 0.35)
            has_demo_video = int(rng.random() < 0.65)
            has_ppt = int(rng.random() < 0.80)
            has_architecture_docs = int(rng.random() < 0.55)
            days_active = int(rng.integers(0, 3))
            template = _MEDIUM_TEMPLATES[int(rng.integers(0, len(_MEDIUM_TEMPLATES)))]
            quality_bonus = -0.55
        else:  # weak
            readme_word_count = int(rng.integers(5, 95))
            file_count = int(rng.integers(1, 7))
            commit_count = int(rng.integers(1, 5))
            contributors_count = 1
            stars_count = 0
            has_tests = int(rng.random() < 0.04)
            has_requirements = int(rng.random() < 0.22)
            has_demo_video = int(rng.random() < 0.28)
            has_ppt = int(rng.random() < 0.20)
            has_architecture_docs = int(rng.random() < 0.12)
            days_active = int(rng.integers(0, 2))
            template = _WEAK_TEMPLATES[int(rng.integers(0, len(_WEAK_TEMPLATES)))]
            quality_bonus = -1.65

        text_summary = template.format(
            name=project_name,
            domain=f"{domain_name} ({domain_terms})",
            tech=tech,
        )

        # Realistic score formulation with nonlinear diminishing returns + human reviewer noise
        eng_depth = (
            0.9 * np.log1p(commit_count)
            + 0.7 * np.log1p(file_count)
            + 1.15 * has_tests
            + 0.75 * has_requirements
            + 0.45 * np.log1p(days_active)
        )
        doc_clarity = (
            0.85 * np.log1p(readme_word_count / 40.0)
            + 1.05 * has_architecture_docs
            + 0.70 * has_demo_video
            + 0.45 * has_ppt
        )

        raw_innovation = 2.1 + 0.55 * eng_depth + 0.45 * doc_clarity + 0.55 * quality_bonus + rng.normal(0.0, 0.72)
        raw_impact = 2.0 + 0.35 * eng_depth + 0.72 * doc_clarity + 0.30 * np.log1p(stars_count) + 0.60 * quality_bonus + rng.normal(0.0, 0.75)

        score_innovation = float(np.round(np.clip(raw_innovation, 1.0, 9.8), 2))
        score_impact = float(np.round(np.clip(raw_impact, 1.0, 9.8), 2))

        # Acceptance probability combines engineering evidence + documentation + stochastic panel noise
        merit_logit = (
            -4.6
            + 0.85 * np.log1p(commit_count)
            + 0.65 * np.log1p(file_count)
            + 0.75 * np.log1p(readme_word_count / 50.0)
            + 1.35 * has_tests
            + 0.90 * has_requirements
            + 0.95 * has_architecture_docs
            + 0.85 * has_demo_video
            + 0.40 * has_ppt
            + 0.85 * quality_bonus
            + rng.normal(0.0, 0.85)
        )
        accepted = int(rng.random() < float(_sigmoid(merit_logit)))

        rows.append(
            {
                "project_name": project_name,
                "domain": domain_name,
                "readme_word_count": readme_word_count,
                "file_count": file_count,
                "commit_count": commit_count,
                "contributors_count": contributors_count,
                "stars_count": stars_count,
                "has_tests": has_tests,
                "has_requirements": has_requirements,
                "has_demo_video": has_demo_video,
                "has_ppt": has_ppt,
                "has_architecture_docs": has_architecture_docs,
                "days_active": days_active,
                "text_summary": text_summary,
                "score_innovation": score_innovation,
                "score_impact": score_impact,
                "accepted": accepted,
            }
        )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    generated = generate_dataset()
    accept_rate = generated["accepted"].mean() * 100.0
    print(f"Generated {len(generated)} rows at {DATA_PATH} (acceptance rate: {accept_rate:.1f}%)")
