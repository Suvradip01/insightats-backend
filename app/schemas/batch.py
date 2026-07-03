from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.analyze import AnalyzeResponse


class RankedResumeResult(BaseModel):
    filename: str
    rank: int = Field(..., ge=1)
    score: int = Field(..., ge=0, le=100)
    ranking_reason: str = ""
    analysis: AnalyzeResponse


class BatchAnalyzeResponse(BaseModel):
    status: str = "success"
    message: Optional[str] = None
    total: int = 0
    results: List[RankedResumeResult] = Field(default_factory=list)
