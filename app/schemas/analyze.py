"""API contract for POST /api/v1/resume/analyze — aligned with InSightATS integration blueprint."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class NerEntities(BaseModel):
    """Raw M1 (BERT NER) output mapped to a stable JSON shape."""

    name: str = ""
    email: str = ""
    skills: List[str] = Field(default_factory=list)
    designation: str = ""
    degree: str = ""
    college_name: str = ""
    companies: List[str] = Field(default_factory=list)
    location: str = ""
    yoe: Optional[str] = None
    grad_year: Optional[str] = None


class BreakdownProbs(BaseModel):
    p_no_fit: float = 0.0
    p_partial: float = 0.0
    p_strong: float = 0.0


class SkillSignals(BaseModel):
    """Derived from M1 skills ∩ JD keywords and YOE vs JD requirements (used with M2 when available)."""

    match: List[str] = Field(default_factory=list)
    exp_gap: int = 0


class FitResult(BaseModel):
    """Raw + derived M2 (RoBERTa matcher) output."""

    label: str = ""
    verdict: str = ""
    fit_score: float = 0.0  # 0.0–1.0
    breakdown: BreakdownProbs = Field(default_factory=BreakdownProbs)
    skill_signals: SkillSignals = Field(default_factory=SkillSignals)
    domain_override: Optional[str] = None


class Confidence3(BaseModel):
    basic: float = 0.0
    intermediate: float = 0.0
    advanced: float = 0.0


class ShapKeywords(BaseModel):
    advanced: List[str] = Field(default_factory=list)
    intermediate: List[str] = Field(default_factory=list)
    basic: List[str] = Field(default_factory=list)


class ProjectComplexity(BaseModel):
    """M3 (DistilBERT complexity) output — or heuristic when the model is not loaded."""

    level: str = "Basic"  # Basic | Intermediate | Advanced
    confidence: Confidence3 = Field(default_factory=Confidence3)
    shap_keywords: ShapKeywords = Field(default_factory=ShapKeywords)
    plain_explanation: str = ""


class AnalyzeResponse(BaseModel):
    status: str
    message: Optional[str] = None

    score: int = Field(
        0,
        ge=0,
        le=100,
        description="Headline match 0–100: M2 fit blended with skill/exp/project/structure axes",
    )
    skill_score: int = Field(0, ge=0, le=100)
    experience_score: int = Field(0, ge=0, le=100)
    project_score: int = Field(0, ge=0, le=100)
    structure_score: int = Field(0, ge=0, le=100)

    feedback: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)

    ner_entities: NerEntities = Field(default_factory=NerEntities)
    fit_result: FitResult = Field(default_factory=FitResult)
    project_complexity: ProjectComplexity = Field(default_factory=ProjectComplexity)

    # Optional explainability (M2 SHAP) — useful for debugging / future UI
    shap_feedback: Optional[str] = None
    raw_shap_data: Dict[str, Any] = Field(default_factory=dict)
