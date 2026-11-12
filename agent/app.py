"""Main Flask orchestration service for the AI-Assisted Mentor Evaluation System."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, scoped_session, sessionmaker

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from github_reader.extractor import extract_github_features  # noqa: E402
from ml.evaluator import ARTIFACTS_DIR, evaluate_submission_ml, get_ml_metrics  # noqa: E402
from video_parser.extractor import extract_video_summary  # noqa: E402

try:
    from cerebras.cloud.sdk import Cerebras  # type: ignore
except ImportError:
    Cerebras = None

GITHUB_READER_URL = os.getenv("GITHUB_READER_URL", "http://localhost:5001")
VIDEO_PARSER_URL = os.getenv("VIDEO_PARSER_URL", "http://localhost:5002")


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class Hackathon(Base):
    __tablename__ = "hackathon"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    judge_id = Column(String(120), nullable=False, index=True)
    criteria = Column(Text, nullable=True)


class Submission(Base):
    __tablename__ = "submission"

    id = Column(Integer, primary_key=True)
    hackathon_id = Column(Integer, ForeignKey("hackathon.id"), nullable=False, index=True)
    project_name = Column(String(150), nullable=False)
    github_link = Column(String(300), nullable=False)
    video_link = Column(String(300), nullable=False)
    ppt_link = Column(String(300), nullable=True)
    status = Column(String(50), default="NEW")
    acceptance_probability = Column(Float, nullable=True)
    score_innovation = Column(Float, nullable=True)
    score_impact = Column(Float, nullable=True)
    positive_points = Column(Text, nullable=True)
    negative_points = Column(Text, nullable=True)
    real_world_necessity = Column(Text, nullable=True)
    ai_generated_percentage = Column(Float, nullable=True)
    mentor_questions = Column(Text, nullable=True)
    justification = Column(Text, nullable=True)
    evaluation_mode = Column(String(100), nullable=True)
    repo_metrics_json = Column(Text, nullable=True)


def _ensure_sqlite_columns(engine) -> None:
    """Safely add newer ML columns if an older hackathon.db file already exists."""
    if not str(engine.url).startswith("sqlite"):
        return
    expected = {
        "acceptance_probability": "FLOAT",
        "evaluation_mode": "VARCHAR(100)",
        "repo_metrics_json": "TEXT",
    }
    with engine.connect() as conn:
        rows = conn.execute(text("PRAGMA table_info(submission)")).fetchall()
        existing = {row[1] for row in rows}
        for col_name, col_type in expected.items():
            if col_name not in existing:
                conn.execute(text(f"ALTER TABLE submission ADD COLUMN {col_name} {col_type}"))
        conn.commit()


def _parse_json_list(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except (json.JSONDecodeError, TypeError):
        return [raw_value]


def _parse_json_dict(raw_value: str | None) -> dict[str, Any]:
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def perform_market_search(query: str) -> str:
    """Execute a lightweight web search check for existing solutions and market need."""
    try:
        clean_query = urllib.parse.quote(query[:120])
        url = f"https://html.duckduckgo.com/html/?q={clean_query}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        html = urllib.request.urlopen(req, timeout=4).read().decode("utf-8", errors="replace")
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
        clean_snippets = [re.sub(r"<[^>]+>", "", s).strip() for s in snippets[:2]]
        if clean_snippets:
            return " | ".join(clean_snippets)
    except Exception:
        pass
    return ""


def _fetch_github_data(github_link: str) -> dict[str, Any]:
    """Query the github-reader microservice if running, falling back to local extraction."""
    if not github_link:
        return {"success": False, "readme_content": "", "repo_features": {}}
    try:
        resp = requests.post(
            f"{GITHUB_READER_URL}/read-repo",
            json={"github_link": github_link},
            timeout=5,
        )
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return extract_github_features(github_link, timeout=6)


def _fetch_video_data(video_link: str) -> dict[str, Any]:
    """Query the video-parser microservice if running, falling back to local extraction."""
    if not video_link:
        return {"success": False, "video_summary": {}}
    try:
        resp = requests.post(
            f"{VIDEO_PARSER_URL}/parse-video",
            json={"video_link": video_link},
            timeout=5,
        )
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return extract_video_summary(video_link, timeout=6)


def _enrich_with_llm(
    api_key: str,
    project_name: str,
    criteria: str,
    readme_text: str,
    video_text: str,
    market_notes: str,
    ml_result: dict[str, Any],
) -> dict[str, Any]:
    """Optionally enrich ML evaluation narratives using Cerebras when an API key is configured."""
    if not api_key or Cerebras is None:
        return ml_result

    prompt = (
        "You are an expert academic & hackathon mentor panelist.\n"
        f"Criteria: {criteria}\n"
        f"Project: {project_name}\n"
        f"ML Model Baseline: decision={ml_result['decision']} "
        f"({ml_result['acceptance_probability']}% confidence), "
        f"innovation={ml_result['score_innovation']}/10, impact={ml_result['score_impact']}/10.\n"
        f"Observed Repo Signals: {json.dumps(ml_result['feature_signals'])}\n"
        f"README Excerpt: {readme_text[:1200]}\n"
        f"Video Info: {video_text[:400]}\n"
        f"Market Notes: {market_notes[:400]}\n\n"
        "Return ONLY valid JSON with keys: positive_points (list of 3 strings), "
        "negative_points (list of 2 strings), mentor_questions (list of 3 strings), "
        "real_world_necessity (string), justification (string)."
    )
    try:
        client = Cerebras(api_key=api_key)
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-4-scout-17b-16e-instruct",
        )
        content = completion.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return ml_result
        llm_data = json.loads(match.group(0))
        enriched = dict(ml_result)
        for key in ("positive_points", "negative_points", "mentor_questions"):
            if isinstance(llm_data.get(key), list) and llm_data[key]:
                enriched[key] = [str(item) for item in llm_data[key][:4]]
        if llm_data.get("real_world_necessity"):
            enriched["real_world_necessity"] = str(llm_data["real_world_necessity"])
        if llm_data.get("justification"):
            enriched["justification"] = str(llm_data["justification"])
        enriched["evaluation_mode"] = f"{ml_result['evaluation_mode']} + Cerebras LLM"
        return enriched
    except Exception:
        return ml_result


def create_app(database_url: str | None = None) -> Flask:
    """Create and configure the Flask application with SQLAlchemy and trained ML inference."""
    load_dotenv()
    app = Flask(__name__, static_folder="static", static_url_path="")

    default_db = f"sqlite:///{(ROOT_DIR / 'agent' / 'hackathon.db').as_posix()}"
    resolved_db_url = database_url or os.getenv("DATABASE_URL") or default_db

    engine_kwargs: dict[str, Any] = {"future": True}
    if resolved_db_url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}

    engine = create_engine(resolved_db_url, **engine_kwargs)
    Base.metadata.create_all(engine)
    _ensure_sqlite_columns(engine)

    SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))

    @app.teardown_appcontext
    def remove_session(_exc=None):
        SessionLocal.remove()

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    @app.route("/")
    def index():
        return app.send_static_file("index.html")

    @app.route("/health", methods=["GET"])
    def health():
        return (
            jsonify(
                {
                    "status": "healthy",
                    "service": "agent",
                    "database": "connected",
                    "ml_ready": (ARTIFACTS_DIR / "acceptance_model.joblib").exists(),
                }
            ),
            200,
        )

    @app.route("/api/ml/metrics", methods=["GET"])
    def ml_metrics():
        try:
            return jsonify(get_ml_metrics()), 200
        except Exception as exc:
            return jsonify({"error": f"Unable to load ML metrics: {exc}"}), 500

    @app.route("/api/ml/charts/<path:filename>", methods=["GET"])
    def ml_chart(filename: str):
        allowed = {
            "model_comparison.png",
            "confusion_matrix.png",
            "feature_importance.png",
            "score_predictions.png",
        }
        if filename not in allowed or not (ARTIFACTS_DIR / filename).exists():
            return jsonify({"error": "Chart not found"}), 404
        return send_from_directory(ARTIFACTS_DIR, filename)

    @app.route("/api/ml/predict", methods=["POST", "OPTIONS"])
    def ml_predict():
        if request.method == "OPTIONS":
            return jsonify(status="ok"), 200
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({"error": "JSON payload is required"}), 400
        result = evaluate_submission_ml(data)
        return jsonify(result), 200

    @app.route("/api/hackathons/create", methods=["POST", "OPTIONS"])
    def create_hackathon():
        if request.method == "OPTIONS":
            return jsonify(status="ok"), 200
        data = request.get_json(silent=True) or {}
        if not data.get("name") or not data.get("judge_id"):
            return jsonify({"error": "Missing name or judge_id"}), 400

        session = SessionLocal()
        new_hackathon = Hackathon(
            name=str(data["name"]).strip(),
            judge_id=str(data["judge_id"]).strip(),
            criteria=str(
                data.get("criteria")
                or "Evaluate based on innovation, real-world impact, technical depth, and documentation."
            ).strip(),
        )
        session.add(new_hackathon)
        session.commit()
        return (
            jsonify(
                {
                    "message": "Hackathon created!",
                    "hackathon": {
                        "id": new_hackathon.id,
                        "name": new_hackathon.name,
                        "criteria": new_hackathon.criteria,
                    },
                }
            ),
            201,
        )

    @app.route("/api/hackathons", methods=["GET"])
    def get_hackathons_by_judge():
        judge_id = request.args.get("judge_id")
        if not judge_id:
            return jsonify({"error": "Judge ID is required"}), 400

        session = SessionLocal()
        hackathons = (
            session.query(Hackathon)
            .filter_by(judge_id=judge_id)
            .order_by(Hackathon.id.desc())
            .all()
        )
        return jsonify(
            [{"id": h.id, "name": h.name, "criteria": h.criteria} for h in hackathons]
        )

    @app.route("/api/hackathons/verify", methods=["POST", "OPTIONS"])
    def verify_hackathon():
        if request.method == "OPTIONS":
            return jsonify(status="ok"), 200
        data = request.get_json(silent=True) or {}
        raw_hid = data.get("hackathon_id")
        if not raw_hid:
            return jsonify({"valid": True, "hackathonName": "Default Session (Auto-assigned)"})
        try:
            hackathon_id = int(raw_hid)
        except (ValueError, TypeError):
            return jsonify({"valid": False, "message": "Invalid Hackathon ID format"}), 400

        session = SessionLocal()
        hackathon = session.get(Hackathon, hackathon_id)
        if hackathon:
            return jsonify({"valid": True, "hackathonName": hackathon.name})
        return jsonify(
            {"valid": False, "message": "Hackathon ID not found (will auto-assign default)"}
        ), 200

    @app.route("/api/dashboard/<int:hackathon_id>", methods=["GET"])
    def dashboard_for_hackathon(hackathon_id: int):
        session = SessionLocal()
        submissions = (
            session.query(Submission)
            .filter_by(hackathon_id=hackathon_id)
            .order_by(Submission.id.desc())
            .all()
        )
        result = []
        for s in submissions:
            result.append(
                {
                    "id": s.id,
                    "project_name": s.project_name,
                    "github_link": s.github_link,
                    "video_link": s.video_link,
                    "ppt_link": s.ppt_link or "",
                    "status": s.status,
                    "acceptance_probability": s.acceptance_probability,
                    "score_innovation": s.score_innovation,
                    "score_impact": s.score_impact,
                    "positive_points": _parse_json_list(s.positive_points),
                    "negative_points": _parse_json_list(s.negative_points),
                    "real_world_necessity": s.real_world_necessity or "",
                    "ai_generated_percentage": (
                        s.ai_generated_percentage if s.ai_generated_percentage is not None else 25.0
                    ),
                    "mentor_questions": _parse_json_list(s.mentor_questions),
                    "justification": s.justification,
                    "evaluation_mode": s.evaluation_mode or "Trained ML Pipeline",
                    "repo_metrics": _parse_json_dict(s.repo_metrics_json),
                }
            )
        return jsonify(result)

    @app.route("/submit", methods=["POST", "OPTIONS"])
    def submit_project():
        if request.method == "OPTIONS":
            return jsonify(status="ok"), 200

        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({"error": "No data provided"}), 400

        github_link = str(data.get("github_link") or "").strip()
        readme_override = str(data.get("readme_text") or data.get("notes") or "").strip()
        if not github_link and not readme_override:
            return jsonify({"error": "Provide either github_link or readme_text"}), 400

        video_link = str(data.get("video_link") or "").strip()
        ppt_link = str(data.get("ppt_link") or "").strip()
        project_name = str(data.get("project_name") or "Untitled Project").strip()

        session = SessionLocal()
        hackathon = None
        raw_hid = data.get("hackathon_id")
        if raw_hid:
            try:
                hackathon = session.get(Hackathon, int(raw_hid))
            except (ValueError, TypeError):
                hackathon = None

        if not hackathon:
            hackathon = session.query(Hackathon).order_by(Hackathon.id.desc()).first()
            if not hackathon:
                hackathon = Hackathon(
                    name="General Mentor Evaluation Session",
                    judge_id="mentor@hackathon.org",
                    criteria="Evaluate based on innovation, real-world impact, technical execution, and documentation.",
                )
                session.add(hackathon)
                session.commit()

        new_submission = Submission(
            hackathon_id=hackathon.id,
            project_name=project_name,
            github_link=github_link or "offline://manual-submission",
            video_link=video_link or "none",
            ppt_link=ppt_link,
            status="PROCESSING",
        )
        session.add(new_submission)
        session.commit()

        try:
            github_info = _fetch_github_data(github_link) if github_link.startswith(("http://", "https://")) else {}
            repo_features = github_info.get("repo_features", {}) if github_info.get("success") else {}
            readme_content = str(github_info.get("readme_content") or "").strip()
            combined_readme = "\n".join(p for p in (readme_content, readme_override) if p).strip()

            video_info = _fetch_video_data(video_link) if video_link.startswith(("http://", "https://")) else {}
            video_summary = video_info.get("video_summary", {}) if video_info.get("success") else {}
            video_text = " ".join(
                str(video_summary.get(k) or "") for k in ("title", "description") if video_summary.get(k) not in (None, "N/A")
            ).strip()

            market_notes = perform_market_search(f"{project_name} project solution")

            ml_payload = {
                "project_name": project_name,
                "readme_text": combined_readme,
                "video_text": video_text,
                "criteria": hackathon.criteria or "",
                "market_search_results": market_notes,
                "has_demo_video": 1 if video_link.startswith(("http://", "https://")) else int(bool(data.get("has_demo_video", 0))),
                "has_ppt": 1 if ppt_link.startswith(("http://", "https://")) else int(bool(data.get("has_ppt", 0))),
            }
            for field in (
                "readme_word_count",
                "file_count",
                "commit_count",
                "contributors_count",
                "stars_count",
                "has_tests",
                "has_requirements",
                "has_architecture_docs",
                "days_active",
            ):
                if data.get(field) is not None and data.get(field) != "":
                    ml_payload[field] = data[field]
            for field, val in repo_features.items():
                if field not in ml_payload:
                    ml_payload[field] = val

            eval_result = evaluate_submission_ml(ml_payload)
            cerebras_key = os.getenv("CEREBRAS_API_KEY", "").strip()
            if cerebras_key:
                eval_result = _enrich_with_llm(
                    cerebras_key,
                    project_name,
                    hackathon.criteria or "",
                    combined_readme,
                    video_text,
                    market_notes,
                    eval_result,
                )

            new_submission.score_innovation = eval_result["score_innovation"]
            new_submission.score_impact = eval_result["score_impact"]
            new_submission.acceptance_probability = eval_result["acceptance_probability"]
            new_submission.positive_points = json.dumps(eval_result["positive_points"])
            new_submission.negative_points = json.dumps(eval_result["negative_points"])
            new_submission.real_world_necessity = eval_result["real_world_necessity"]
            new_submission.ai_generated_percentage = eval_result["ai_generated_percentage"]
            new_submission.mentor_questions = json.dumps(eval_result["mentor_questions"])
            new_submission.justification = eval_result["justification"]
            new_submission.evaluation_mode = eval_result["evaluation_mode"]
            new_submission.repo_metrics_json = json.dumps(eval_result["feature_signals"])
            new_submission.status = (
                "AI_ACCEPTED" if eval_result["decision"] == "ACCEPTED" else "AI_REJECTED"
            )
            session.commit()

            return (
                jsonify(
                    {
                        "message": "Project submitted & evaluated by ML pipeline!",
                        "submission_id": new_submission.id,
                        "hackathon_id": hackathon.id,
                        "status": new_submission.status,
                        "evaluation": eval_result,
                    }
                ),
                201,
            )
        except Exception as exc:
            session.rollback()
            new_submission.status = "SYSTEM_ERROR"
            new_submission.justification = f"Evaluation failed: {exc}"
            session.commit()
            return jsonify({"error": f"Evaluation failed: {exc}"}), 500

    return app


if __name__ == "__main__":
    application = create_app()
    port = int(os.getenv("PORT", "5000"))
    application.run(host="0.0.0.0", port=port, debug=False)
