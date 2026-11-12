"""Automated unit and integration tests for the ML pipeline, feature extraction, and Flask API."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.app import create_app
from github_reader.extractor import parse_github_repo_path
from ml.evaluator import evaluate_submission_ml, get_ml_metrics
from ml.features import build_feature_row, extract_text_signals


class TestFeatureAndGitHubParsing(unittest.TestCase):
    def test_parse_github_repo_path_variants(self):
        self.assertEqual(
            parse_github_repo_path("https://github.com/Tushar-Kosish/AI-Asisted-Mentor-Evaluating-system.git"),
            "Tushar-Kosish/AI-Asisted-Mentor-Evaluating-system",
        )
        self.assertEqual(
            parse_github_repo_path("https://github.com/org/repo/tree/main"),
            "org/repo",
        )
        self.assertIsNone(parse_github_repo_path("https://example.com/not-github"))
        self.assertIsNone(parse_github_repo_path("invalid-url"))

    def test_extract_text_signals_and_feature_row(self):
        readme = (
            "## Problem Statement\n"
            "System architecture and REST API pipeline.\n"
            "## Setup\nRun python run_local.py to test."
        )
        signals = extract_text_signals(readme)
        self.assertGreaterEqual(signals["readme_word_count"], 12)
        self.assertEqual(signals["has_architecture_docs"], 1)

        row = build_feature_row(
            {
                "project_name": "Demo",
                "readme_text": readme,
                "commit_count": 20,
                "has_tests": True,
            }
        )
        self.assertEqual(row["commit_count"], 20)
        self.assertEqual(row["has_tests"], 1)
        self.assertIn("Demo", row["text_summary"])


class TestMLPipeline(unittest.TestCase):
    def test_metrics_and_artifacts_present(self):
        metrics = get_ml_metrics()
        self.assertIn("classification", metrics)
        self.assertIn("regression_innovation", metrics)
        self.assertIn("regression_impact", metrics)
        self.assertGreaterEqual(metrics["dataset"]["total_rows"], 500)

    def test_strong_submission_outscores_weak_submission(self):
        strong = evaluate_submission_ml(
            {
                "project_name": "AgriPulse Crop Disease Classifier",
                "readme_text": (
                    "End-to-end AgriTech ML platform using scikit-learn, TF-IDF, and Flask API. "
                    "Includes architecture diagram, installation setup, cross-validation metrics, "
                    "and automated tests for deployment."
                ),
                "readme_word_count": 420,
                "file_count": 32,
                "commit_count": 40,
                "contributors_count": 3,
                "stars_count": 8,
                "has_tests": 1,
                "has_requirements": 1,
                "has_demo_video": 1,
                "has_ppt": 1,
                "has_architecture_docs": 1,
                "days_active": 18,
            }
        )
        weak = evaluate_submission_ml(
            {
                "project_name": "Draft Idea",
                "readme_text": "Initial commit placeholder readme work in progress.",
                "readme_word_count": 12,
                "file_count": 2,
                "commit_count": 1,
                "contributors_count": 1,
                "stars_count": 0,
                "has_tests": 0,
                "has_requirements": 0,
                "has_demo_video": 0,
                "has_ppt": 0,
                "has_architecture_docs": 0,
                "days_active": 0,
            }
        )

        self.assertEqual(strong["decision"], "ACCEPTED")
        self.assertEqual(weak["decision"], "REJECTED")
        self.assertGreater(strong["acceptance_probability"], weak["acceptance_probability"])
        self.assertGreater(strong["score_innovation"], weak["score_innovation"])
        self.assertGreater(strong["score_impact"], weak["score_impact"])
        self.assertTrue(len(strong["positive_points"]) >= 1)
        self.assertTrue(len(weak["negative_points"]) >= 1)
        self.assertEqual(len(weak["mentor_questions"]), 3)


class TestFlaskAPI(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.tmp_dir.name) / "test.db"
        self.app = create_app(database_url=f"sqlite:///{db_path.as_posix()}")
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_health_and_ml_endpoints(self):
        health_resp = self.client.get("/health")
        self.assertEqual(health_resp.status_code, 200)
        self.assertTrue(health_resp.get_json()["ml_ready"])

        metrics_resp = self.client.get("/api/ml/metrics")
        self.assertEqual(metrics_resp.status_code, 200)
        self.assertIn("classification", metrics_resp.get_json())

        chart_resp = self.client.get("/api/ml/charts/confusion_matrix.png")
        self.assertEqual(chart_resp.status_code, 200)
        self.assertEqual(chart_resp.mimetype, "image/png")

    def test_end_to_end_session_submission_and_dashboard(self):
        create_resp = self.client.post(
            "/api/hackathons/create",
            json={
                "name": "Academic Review 2026",
                "judge_id": "prof@college.edu",
                "criteria": "Innovation, architecture, testing, and impact.",
            },
        )
        self.assertEqual(create_resp.status_code, 201)
        hack_id = create_resp.get_json()["hackathon"]["id"]

        verify_resp = self.client.post("/api/hackathons/verify", json={"hackathon_id": hack_id})
        self.assertEqual(verify_resp.status_code, 200)
        self.assertTrue(verify_resp.get_json()["valid"])

        submit_resp = self.client.post(
            "/submit",
            json={
                "hackathon_id": hack_id,
                "project_name": "EduMentor ML",
                "readme_text": (
                    "Complete EdTech machine learning pipeline with TF-IDF, RandomForest, "
                    "architecture docs, setup guide, evaluation metrics, and automated unit tests."
                ),
                "readme_word_count": 320,
                "commit_count": 25,
                "file_count": 24,
                "has_tests": 1,
                "has_requirements": 1,
                "has_demo_video": 1,
                "days_active": 14,
            },
        )
        self.assertEqual(submit_resp.status_code, 201)
        payload = submit_resp.get_json()
        self.assertEqual(payload["status"], "AI_ACCEPTED")
        self.assertGreaterEqual(payload["evaluation"]["acceptance_probability"], 70.0)

        dash_resp = self.client.get(f"/api/dashboard/{hack_id}")
        self.assertEqual(dash_resp.status_code, 200)
        items = dash_resp.get_json()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["project_name"], "EduMentor ML")
        self.assertIn("commit_count", items[0]["repo_metrics"])


if __name__ == "__main__":
    unittest.main()
