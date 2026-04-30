import json
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.schemas.batch import BatchAnalyzeResponse, RankedResumeResult
from app.schemas.job import JobDescription
from app.schemas.recruiter import (
    RecruiterAuthResponse,
    RecruiterLoginRequest,
    RecruiterRegisterRequest,
)
from app.services.pipeline.orchestrator import get_orchestrator
from app.services.recruiter.security import AuthPrincipal, hash_password, verify_password
from app.services.recruiter.store import create_recruiter, create_session, get_recruiter_by_company_username, get_recruiter_by_username
from app.services.resume_parser import ResumeParser
import base64

router = APIRouter()
_bearer = HTTPBearer(auto_error=False)


from fastapi import Request

def _require_recruiter(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> AuthPrincipal:
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    
    token = creds.credentials
    try:
        parts = token.split('.')
        if len(parts) != 3:
            raise ValueError("Invalid JWT")
        
        payload_b64 = parts[1]
        payload_b64 += '=' * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode('utf-8'))
        
        rid = payload.get("sub", "unknown")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    company = request.headers.get("x-company-name", "your company")
    return AuthPrincipal(recruiter_id=0, company=company, username=payload.get("email", "unknown"))


@router.post("/register", response_model=RecruiterAuthResponse)
def register_recruiter(payload: RecruiterRegisterRequest) -> RecruiterAuthResponse:
    existing = get_recruiter_by_company_username(payload.company, payload.username)
    if existing:
        raise HTTPException(status_code=409, detail="Recruiter already exists for this company/username")
    ph = hash_password(payload.password)
    try:
        rid = create_recruiter(payload.company, payload.username, ph)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    token = create_session(rid)
    return RecruiterAuthResponse(token=token)


@router.post("/login", response_model=RecruiterAuthResponse)
def login_recruiter(payload: RecruiterLoginRequest) -> RecruiterAuthResponse:
    row = get_recruiter_by_username(payload.username)
    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(payload.password, str(row["password_hash"])):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_session(int(row["id"]))
    return RecruiterAuthResponse(token=token)


def _ranking_reason(analysis) -> str:
    # Use the most explainable signals already computed.
    miss = len(getattr(analysis, "missing_skills", []) or [])
    fit = getattr(getattr(analysis, "fit_result", None), "verdict", "") or ""
    return (
        f"Overall score {analysis.score}/100. "
        f"Fit verdict: {fit or 'N/A'}. "
        f"Missing skills: {miss}. "
        f"Skills/Exp/Projects/Structure = {analysis.skill_score}/{analysis.experience_score}/{analysis.project_score}/{analysis.structure_score}."
    )


@router.post("/batch-analyze", response_model=BatchAnalyzeResponse)
async def batch_analyze(
    resumes: List[UploadFile] = File(..., description="Multiple PDF/DOCX/TXT resumes"),
    job_description_file: UploadFile = File(..., description="Job description PDF/DOCX/TXT file"),
    recruiter: AuthPrincipal = Depends(_require_recruiter),
) -> BatchAnalyzeResponse:
    try:
        jd_content = await job_description_file.read()
        jd_filename = job_description_file.filename or "jd.txt"
        jd_text = ResumeParser.extract_text(jd_content, jd_filename)
        job = JobDescription(title="Target Role", description=jd_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid job_description file: {e}") from e

    orch = get_orchestrator()
    analyzed = []

    for rf in resumes:
        content = await rf.read()
        filename = rf.filename or "resume.bin"
        try:
            text = ResumeParser.extract_text(content, filename)
            analysis = orch.analyze(text, job)
            analyzed.append((filename, analysis))
        except ValueError as ve:
            # Keep failed file in results with score 0 + message so recruiters can spot issues.
            from app.schemas.analyze import AnalyzeResponse

            analyzed.append(
                (
                    filename,
                    AnalyzeResponse(
                        status="error",
                        message=str(ve),
                        score=0,
                        skill_score=0,
                        experience_score=0,
                        project_score=0,
                        structure_score=0,
                        feedback=[],
                        missing_skills=[],
                    ),
                )
            )

    analyzed.sort(key=lambda t: int(getattr(t[1], "score", 0) or 0), reverse=True)

    results: List[RankedResumeResult] = []
    for idx, (filename, analysis) in enumerate(analyzed, start=1):
        results.append(
            RankedResumeResult(
                filename=filename,
                rank=idx,
                score=int(analysis.score or 0),
                ranking_reason=_ranking_reason(analysis),
                analysis=analysis,
            )
        )

    return BatchAnalyzeResponse(
        status="success",
        message=f"Analyzed {len(results)} resume(s) for {recruiter.company}.",
        total=len(results),
        results=results,
    )
