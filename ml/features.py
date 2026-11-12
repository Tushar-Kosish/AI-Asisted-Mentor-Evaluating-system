"""Shared feature definitions and extraction helpers for training and inference."""

from __future__ import annotations

import re
from typing import Any

NUMERIC_FEATURES: list[str] = [
    "readme_word_count",
    "file_count",
    "commit_count",
    "contributors_count",
    "stars_count",
    "has_tests",
    "has_requirements",
    "has_demo_video",
    "has_ppt",
    "has_architecture_docs",
    "days_active",
]

TEXT_COLUMN = "text_summary"

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "readme_word_count": "README Word Count",
    "file_count": "Repository File Count",
    "commit_count": "Commit Count",
    "contributors_count": "Contributors",
    "stars_count": "GitHub Stars",
    "has_tests": "Automated Tests Present",
    "has_requirements": "Dependency Manifest Present",
    "has_demo_video": "Pitch Video Provided",
    "has_ppt": "Presentation Deck Provided",
    "has_architecture_docs": "Architecture / Setup Docs",
    "days_active": "Active Development Days",
}

_DOC_SECTION_KEYWORDS = (
    "architecture",
    "setup",
    "installation",
    "install",
    "usage",
    "how to run",
    "methodology",
    "pipeline",
    "problem statement",
    "evaluation",
    "api",
    "tech stack",
)


def extract_text_signals(text: str) -> dict[str, int]:
    """Derive word count and documentation completeness signals from project text."""
    cleaned = (text or "").strip()
    words = re.findall(r"\b\w+\b", cleaned)
    lowered = cleaned.lower()
    has_arch = int(any(kw in lowered for kw in _DOC_SECTION_KEYWORDS))
    return {
        "readme_word_count": len(words),
        "has_architecture_docs": has_arch,
    }


def _to_int(value: Any, default: int, minimum: int = 0, maximum: int = 100000) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _to_binary(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return 1
        if normalized in {"0", "false", "no", "n", ""}:
            return 0
    return default


def build_feature_row(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw project/GitHub metadata and text into the model input schema."""
    project_name = str(payload.get("project_name") or "").strip()
    readme_text = str(payload.get("readme_text") or payload.get("readme_content") or "").strip()
    video_text = str(payload.get("video_text") or "").strip()
    criteria_text = str(payload.get("criteria") or "").strip()

    combined_doc = " ".join(part for part in (readme_text, video_text) if part).strip()
    text_signals = extract_text_signals(readme_text or combined_doc or project_name)

    readme_word_count = _to_int(
        payload.get("readme_word_count"),
        default=text_signals["readme_word_count"],
        minimum=0,
        maximum=20000,
    )
    has_architecture_docs = _to_binary(
        payload.get("has_architecture_docs"),
        default=text_signals["has_architecture_docs"],
    )

    text_summary = " ".join(
        part for part in (project_name, readme_text, video_text, criteria_text) if part
    ).strip()
    if not text_summary:
        text_summary = "software prototype submission"

    return {
        "readme_word_count": readme_word_count,
        "file_count": _to_int(payload.get("file_count"), default=8, minimum=0, maximum=50000),
        "commit_count": _to_int(payload.get("commit_count"), default=5, minimum=0, maximum=50000),
        "contributors_count": _to_int(payload.get("contributors_count"), default=1, minimum=1, maximum=500),
        "stars_count": _to_int(payload.get("stars_count"), default=0, minimum=0, maximum=100000),
        "has_tests": _to_binary(payload.get("has_tests"), default=0),
        "has_requirements": _to_binary(payload.get("has_requirements"), default=1 if readme_word_count > 60 else 0),
        "has_demo_video": _to_binary(payload.get("has_demo_video"), default=0),
        "has_ppt": _to_binary(payload.get("has_ppt"), default=0),
        "has_architecture_docs": has_architecture_docs,
        "days_active": _to_int(payload.get("days_active"), default=3, minimum=0, maximum=3650),
        TEXT_COLUMN: text_summary,
    }
