"""End-to-end InSightATS pipeline: M1 → M2 → M3 → derived scores → feedback."""

from __future__ import annotations

from app.core.config import settings
from app.schemas.analyze import AnalyzeResponse, BreakdownProbs, FitResult, SkillSignals
from app.schemas.job import JobDescription
from app.services.feedback.builder import build_feedback_lines
from app.services.inference.complexity import ComplexityRunner
from app.services.inference.matcher import MatcherRunner
from app.services.inference.ner import NerRunner
from app.services.scoring.derive_scores import (
    compute_exp_gap_years,
    derive_experience_score,
    derive_skill_score,
    expand_skill_phrases,
    extract_jd_required_years,
    extract_resume_yoe_years,
    infer_skills_from_resume_text,
    jd_required_skills,
    project_score_from_level,
    refine_overlap_with_resume_text,
    structure_score_from_entities,
)


def _composite_headline_score(
    fit_score_0_1: float,
    skill_score: int,
    experience_score: int,
    project_score: int,
    structure_score: int,
) -> int:
    """
    Blend M2 semantic match with the four radar dimensions.
    
    Using only fit_score × 100 for the donut often under-reports strong JD/CV overlap when the
    classifier still favors "Partial Fit" — the radar then looks good while the headline stays ~60s.
    """
    m2 = fit_score_0_1 * 100.0
    dim = (
        0.35 * skill_score
        + 0.25 * experience_score
        + 0.25 * project_score
        + 0.15 * structure_score
    )
    # The user's M2 semantic ML model dictates 42% of the total score directly
    blended = 0.42 * m2 + 0.58 * dim
    return int(round(max(0.0, min(100.0, blended))))


def _synthetic_fit(skill_signals: SkillSignals, jd_skill_count: int) -> FitResult:
    """When M2 is unavailable, approximate fit_score from skill overlap density."""
    m = len(skill_signals.match)
    raw = 0.15 + 0.85 * min(1.0, m / max(jd_skill_count, 1))
    raw = max(0.0, min(1.0, raw))

    return FitResult(
        label="Partial Fit",
        verdict="PARTIAL FIT" if raw >= 0.35 else "NOT A FIT",
        fit_score=raw,
        breakdown=BreakdownProbs(
            p_no_fit=round(1.0 - raw, 4),
            p_partial=round(0.6 * raw, 4),
            p_strong=round(0.4 * raw, 4),
        ),
        skill_signals=skill_signals,
        domain_override="matcher_unavailable",
    )


class InsightOrchestrator:
    def __init__(self) -> None:
        self.ner = NerRunner()
        self.matcher = MatcherRunner()
        self.complexity = ComplexityRunner()

    def analyze(self, resume_text: str, job: JobDescription) -> AnalyzeResponse:
        jd_text = job.description or ""

        if not self.ner.loaded and not self.matcher.loaded:
            return AnalyzeResponse(
                status="pending_setup",
                message=(
                    "No models loaded. Add ner_model to backend/models/ner_model/ "
                    "and/or matcher_model to backend/models/matcher_model/, then restart."
                ),
                score=0,
                feedback=[],
                shap_feedback="Place model weights under backend/models/ and restart the server.",
            )

        grouped = self.ner.extract_grouped(resume_text) if self.ner.loaded else {}
        ner_entities = NerRunner.to_contract(grouped, resume_text)

        resume_skills = expand_skill_phrases(list(ner_entities.skills))
        if not resume_skills:
            resume_skills = infer_skills_from_resume_text(resume_text)
        if resume_skills:
            ner_entities = ner_entities.model_copy(update={"skills": resume_skills})

        jd_skills = jd_required_skills(jd_text, job.mandatory_skills, job.preferred_skills)
        matched, missing = refine_overlap_with_resume_text(resume_text, resume_skills, jd_skills)

        resume_for_m2 = (
            "Skills identified (M1 + resume text): "
            + ", ".join(resume_skills[:120])
            + "\n\n"
            + resume_text
        )[:4500]

        resume_yoe = extract_resume_yoe_years(resume_text, ner_entities.yoe)
        jd_need_years = extract_jd_required_years(jd_text)
        exp_gap = compute_exp_gap_years(resume_yoe, jd_need_years)

        skill_signals = SkillSignals(match=matched, exp_gap=exp_gap)

        if self.matcher.loaded:
            fit_result = self.matcher.build_fit_result(
                resume_text, jd_text, skill_signals, resume_for_encoding=resume_for_m2
            )
            if settings.ENABLE_SHAP:
                shap_msg, tokens, values = self.matcher.shap_explain(
                    resume_text, jd_text, resume_for_encoding=resume_for_m2
                )
                raw_shap = {"tokens": tokens, "values": values}
            else:
                shap_msg = ""
                raw_shap = {"tokens": [], "values": []}
        else:
            fit_result = _synthetic_fit(skill_signals, max(len(jd_skills), 1))
            shap_msg = "Matcher not loaded — fit_score synthesized from skill overlap."
            raw_shap = {"tokens": [], "values": []}

        proj = self.complexity.predict(resume_text)

        structure = structure_score_from_entities(grouped or {}, resume_text)
        proj_score = project_score_from_level(proj.level)

        fs = fit_result.fit_score
        skill_score = derive_skill_score(len(matched), max(len(jd_skills), 1), fs)
        experience_score = derive_experience_score(resume_yoe, jd_need_years, exp_gap, fs)

        if self.matcher.loaded:
            overall = _composite_headline_score(
                fs, skill_score, experience_score, proj_score, structure
            )
        else:
            overall = int(
                round(
                    0.35 * skill_score
                    + 0.25 * experience_score
                    + 0.25 * proj_score
                    + 0.15 * structure
                )
            )

        feedback = build_feedback_lines(
            skill_score=skill_score,
            missing_skills=missing,
            experience_score=experience_score,
            project=proj,
            fit=fit_result,
            shap_feedback=shap_msg,
        )

        return AnalyzeResponse(
            status="success",
            score=max(0, min(100, overall)),
            skill_score=skill_score,
            experience_score=experience_score,
            project_score=proj_score,
            structure_score=structure,
            feedback=feedback,
            missing_skills=missing,
            ner_entities=ner_entities,
            fit_result=fit_result,
            project_complexity=proj,
            shap_feedback=shap_msg,
            raw_shap_data=raw_shap,
        )


_orchestrator: InsightOrchestrator | None = None


def get_orchestrator() -> InsightOrchestrator:
    """Lazy singleton — loads transformers on first analyze (faster API startup)."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = InsightOrchestrator()
    return _orchestrator
