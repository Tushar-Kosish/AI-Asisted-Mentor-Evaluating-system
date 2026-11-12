"""Shared GitHub repository metadata, README, and engineering signal extractor."""

from __future__ import annotations

import base64
import os
import re
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import requests

_DEPENDENCY_FILES = {
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "package.json",
    "environment.yml",
    "pipfile",
    "pom.xml",
    "build.gradle",
    "cargo.toml",
    "go.mod",
}


def parse_github_repo_path(github_link: str) -> str | None:
    """Extract 'owner/repo' from common GitHub URL formats."""
    if not github_link or not isinstance(github_link, str):
        return None
    raw = github_link.strip()
    if not raw.startswith(("http://", "https://")):
        return None
    parsed = urlparse(raw)
    if "github.com" not in parsed.netloc.lower():
        return None
    segments = [seg for seg in parsed.path.split("/") if seg]
    if len(segments) < 2:
        return None
    owner, repo = segments[0], segments[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        return None
    return f"{owner}/{repo}"


def _parse_last_page_count(link_header: str) -> int | None:
    """Parse total item count from GitHub's 'Link' header when per_page=1."""
    if not link_header:
        return None
    match = re.search(r'[?&]page=(\d+)>;\s*rel="last"', link_header)
    if match:
        return int(match.group(1))
    return None


def _compute_days_active(created_at: str | None, pushed_at: str | None) -> int:
    if not created_at or not pushed_at:
        return 1
    try:
        start = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
        return max(1, (end - start).days)
    except ValueError:
        return 1


def extract_github_features(
    github_link: str,
    token: str | None = None,
    timeout: int = 8,
) -> dict[str, Any]:
    """Fetch README and repository engineering signals from the GitHub REST API."""
    repo_path = parse_github_repo_path(github_link)
    if not repo_path:
        return {
            "success": False,
            "error": "Invalid GitHub repository URL.",
            "readme_content": "",
            "repo_features": {},
        }

    headers = {"Accept": "application/vnd.github.v3+json"}
    auth_token = token or os.getenv("GITHUB_TOKEN")
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    base_url = f"https://api.github.com/repos/{repo_path}"

    try:
        repo_resp = requests.get(base_url, headers=headers, timeout=timeout)
        repo_resp.raise_for_status()
        repo_meta = repo_resp.json()
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "readme_content": "",
            "repo_features": {},
        }

    default_branch = repo_meta.get("default_branch", "main")
    stars_count = int(repo_meta.get("stargazers_count") or 0)
    days_active = _compute_days_active(repo_meta.get("created_at"), repo_meta.get("pushed_at"))
    description = str(repo_meta.get("description") or "").strip()

    # Fetch README using GitHub's canonical /readme endpoint (case-insensitive)
    readme_content = ""
    try:
        readme_resp = requests.get(f"{base_url}/readme", headers=headers, timeout=timeout)
        if readme_resp.ok:
            encoded = readme_resp.json().get("content", "")
            if encoded:
                readme_content = base64.b64decode(encoded).decode("utf-8", errors="replace")
    except Exception:
        readme_content = ""

    if not readme_content and description:
        readme_content = description

    # Inspect repository file tree for file count, tests, and dependency manifests
    file_count = 0
    has_tests = 0
    has_requirements = 0
    try:
        tree_resp = requests.get(
            f"{base_url}/git/trees/{default_branch}?recursive=1",
            headers=headers,
            timeout=timeout,
        )
        if tree_resp.ok:
            blobs = [
                item.get("path", "")
                for item in tree_resp.json().get("tree", [])
                if item.get("type") == "blob"
            ]
            file_count = len(blobs)
            for path in blobs:
                lower = path.lower()
                basename = lower.rsplit("/", 1)[-1]
                if (
                    "/test" in lower
                    or lower.startswith("test")
                    or "tests/" in lower
                    or "_test." in basename
                    or ".test." in basename
                    or ".spec." in basename
                ):
                    has_tests = 1
                if basename in _DEPENDENCY_FILES:
                    has_requirements = 1
    except Exception:
        pass

    # Estimate commit count efficiently via per_page=1 Link header
    commit_count = 1
    try:
        commits_resp = requests.get(
            f"{base_url}/commits?per_page=1",
            headers=headers,
            timeout=timeout,
        )
        if commits_resp.ok:
            parsed_total = _parse_last_page_count(commits_resp.headers.get("Link", ""))
            if parsed_total is not None:
                commit_count = parsed_total
            elif isinstance(commits_resp.json(), list):
                commit_count = max(1, len(commits_resp.json()))
    except Exception:
        pass

    # Estimate contributors count
    contributors_count = 1
    try:
        contrib_resp = requests.get(
            f"{base_url}/contributors?per_page=20&anon=1",
            headers=headers,
            timeout=timeout,
        )
        if contrib_resp.ok and isinstance(contrib_resp.json(), list):
            contributors_count = max(1, len(contrib_resp.json()))
    except Exception:
        pass

    return {
        "success": True,
        "repo_path": repo_path,
        "readme_content": readme_content,
        "repo_features": {
            "file_count": file_count,
            "commit_count": commit_count,
            "contributors_count": contributors_count,
            "stars_count": stars_count,
            "has_tests": has_tests,
            "has_requirements": has_requirements,
            "days_active": days_active,
            "language": repo_meta.get("language") or "Unknown",
        },
    }
