"""Human-readable feedback lines — tuned for `parseFeedback()` in Dashboard.jsx."""

from __future__ import annotations

from typing import List

from app.schemas.analyze import FitResult, ProjectComplexity


def _level_phrase(level: str) -> str:
    lv = (level or "Basic").strip().lower()
    if lv.startswith("adv"):
        return "Advanced Level"
    if lv.startswith("inter") or lv.startswith("medium"):
        return "Medium Level"
    return "Basic Level"


def build_feedback_lines(
    *,
    skill_score: int,
    missing_skills: List[str],
    experience_score: int,
    project: ProjectComplexity,
    fit: FitResult,
    shap_feedback: str,
) -> List[str]:
    lines: List[str] = []

    if skill_score >= 80 and not missing_skills:
        lines.append(
            "✅ **Skills**: All mandatory skills detected based on job description keywords."
        )
    else:
        miss = ", ".join(missing_skills[:12]) if missing_skills else "key relevant technologies"
        lines.append(f"❌ **Critical Missing Skills**: You might be missing: {miss}")

    lvl = _level_phrase(project.level)
    if lvl == "Advanced Level":
        lines.append(
            f"**Project Detected ({lvl})**: {project.plain_explanation}"
        )
    elif lvl == "Medium Level":
        lines.append(
            f"**Project Detected ({lvl})**: {project.plain_explanation}"
        )
    else:
        lines.append(
            f"**Project Detected ({lvl})**: Consider adding metrics, stack details, and impact."
        )
        lines.append(
            "Tip for Project: Strengthen descriptions with technologies used and measurable outcomes."
        )

    if experience_score >= 75:
        lines.append(
            "**Experience Relevance**: Your experience section aligns well with typical role expectations."
        )
    else:
        lines.append(
            "**Experience Relevance**: Clarify roles, timelines, and outcomes; align bullets with the JD."
        )

    # verdict is kept in API for compatibility; label is the human-readable class name.
    overall = f"**Overall**: {fit.label}."
    if shap_feedback and shap_feedback.strip():
        overall = f"{overall} {shap_feedback.strip()}"
    lines.append(overall)

    return lines
