"""
Recruiter API endpoints — register, login, and batch resume analysis.

Auth strategy
-------------
Two token types are accepted for protected endpoints:
  1. Clerk JWT  (3-part dot-separated, signed RS256) — used by the React app
     via useAuth().getToken(). Requires CLERK_ISSUER to be configured on the
     backend. If it is not configured, the endpoint returns HTTP 503 with a
     clear message instead of a cryptic "CLERK_ISSUER is not configured" 401.
  2. SQLite session token — opaque hex string issued on /login. Used as a
     fallback if the frontend is not using Clerk (e.g. self-hosted mode).

Batch analysis concurrency
--------------------------
Processing 100 resumes sequentially took ~10-30 min and often timed out.
Each `orchestrator.analyze()` call is CPU-bound (transformers inference) and
runs synchronously.  We run them concurrently in a ThreadPoolExecutor so all
resumes are processed in parallel, bounded by MAX_BATCH_WORKERS (default 4
to avoid OOM on CPU-only servers). On a local GPU/CPU machine with more RAM
you can raise this via the BATCH_WORKERS env variable.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.schemas.batch import BatchAnalyzeResponse, RankedResumeResult
from app.schemas.job import JobDescription
from app.schemas.analyze import AnalyzeResponse
from app.services.pipeline.orchestrator import get_orchestrator
from app.services.recruiter.clerk_jwt import _looks_like_jwt, verify_clerk_jwt
from app.services.recruiter.security import AuthPrincipal, hash_password, verify_password
from app.services.recruiter.store import (
    create_recruiter,
    create_session,
    get_principal_for_token,
    get_recruiter_by_company_username,
    get_recruiter_by_username,
)
from app.services.resume_parser import ResumeParser
from app.schemas.recruiter import (
    RecruiterAuthResponse,
    RecruiterLoginRequest,
    RecruiterRegisterRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()
_bearer = HTTPBearer(auto_error=False)

# Maximum parallel threads for batch analysis.
# Raise via BATCH_WORKERS env var if your server has more RAM/CPU headroom.
MAX_BATCH_WORKERS: int = int(os.environ.get("BATCH_WORKERS", "4"))

# Shared executor — created once, reused across requests.
_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=MAX_BATCH_WORKERS, thread_name_prefix="batch-worker")
    return _executor


# ── Auth helpers ──────────────────────────────────────────────────────────────


def _principal_from_clerk_payload(payload: dict, request: Request) -> AuthPrincipal:
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Invalid token: missing subject")
    email = (
        payload.get("email")
        or payload.get("primary_email")
        or payload.get("username")
        or str(sub)
    )
    company = request.headers.get("x-company-name") or "your company"
    return AuthPrincipal(recruiter_id=0, company=company, username=str(email))


def _principal_from_session_token(token: str) -> AuthPrincipal:
    row = get_principal_for_token(token)
    if not row:
        raise HTTPException(status_code=401, detail="Invalid or expired session token")
    rid, company, username = row
    return AuthPrincipal(recruiter_id=rid, company=company, username=username)


def _require_recruiter(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> AuthPrincipal:
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = creds.credentials.strip()

    if _looks_like_jwt(token):
        # Clerk JWT path
        try:
            payload = verify_clerk_jwt(token)
        except RuntimeError as e:
            # CLERK_ISSUER / CLERK_JWKS_URL not configured on the backend.
            raise HTTPException(
                status_code=503,
                detail=(
                    "Clerk JWT verification is not configured on this server. "
                    "Set CLERK_ISSUER in the backend environment variables. "
                    f"Detail: {e}"
                ),
            ) from e
        except ValueError as e:
            raise HTTPException(status_code=401, detail=str(e)) from e
        return _principal_from_clerk_payload(payload, request)

    # Opaque session token path (fallback / self-hosted)
    return _principal_from_session_token(token)


# ── Auth endpoints ────────────────────────────────────────────────────────────


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


# ── Batch analysis ────────────────────────────────────────────────────────────


def _ranking_reason(analysis: AnalyzeResponse) -> str:
    miss = len(getattr(analysis, "missing_skills", []) or [])
    fit = getattr(getattr(analysis, "fit_result", None), "verdict", "") or ""
    return (
        f"Overall score {analysis.score}/100. "
        f"Fit verdict: {fit or 'N/A'}. "
        f"Missing skills: {miss}. "
        f"Skills/Exp/Projects/Structure = "
        f"{analysis.skill_score}/{analysis.experience_score}/{analysis.project_score}/{analysis.structure_score}."
    )


def _analyze_one(filename: str, content: bytes, job: JobDescription) -> tuple[str, AnalyzeResponse]:
    """
    Parse + analyze a single resume synchronously (runs in a thread worker).

    Returns (filename, AnalyzeResponse).  Never raises — errors are captured
    as a status="error" AnalyzeResponse so one bad resume does not abort the batch.
    """
    try:
        text = ResumeParser.extract_text(content, filename)
        result = get_orchestrator().analyze(text, job)
        return filename, result
    except ValueError as ve:
        logger.warning("Resume parse error (%s): %s", filename, ve)
        return filename, AnalyzeResponse(
            status="error",
            message=str(ve),
            score=0,
            skill_score=0,
            experience_score=0,
            project_score=0,
            structure_score=0,
            feedback=[],
            missing_skills=[],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error analyzing %s: %s", filename, exc, exc_info=True)
        return filename, AnalyzeResponse(
            status="error",
            message=f"Unexpected error: {exc}",
            score=0,
            skill_score=0,
            experience_score=0,
            project_score=0,
            structure_score=0,
            feedback=[],
            missing_skills=[],
        )


@router.post("/batch-analyze", response_model=BatchAnalyzeResponse)
async def batch_analyze(
    resumes: List[UploadFile] = File(..., description="Multiple PDF/DOCX/TXT resumes"),
    job_description_file: UploadFile = File(..., description="Job description PDF/DOCX/TXT file"),
    recruiter: AuthPrincipal = Depends(_require_recruiter),
) -> BatchAnalyzeResponse:
    """
    Analyze multiple resumes against a job description concurrently.

    Each resume is processed in a thread-pool worker (CPU-bound inference),
    while the event loop stays free. Results are ranked by score descending.

    Concurrency is bounded by BATCH_WORKERS (default 4). Raise this via the
    environment variable if your server has more headroom.
    """
    # ── Read JD file ──────────────────────────────────────────────────────
    try:
        jd_content = await job_description_file.read()
        jd_filename = job_description_file.filename or "jd.txt"
        jd_text = ResumeParser.extract_text(jd_content, jd_filename)
        job = JobDescription(title="Target Role", description=jd_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid job_description file: {e}") from e

    # ── Read all resume bytes up-front (async, fast) ──────────────────────
    resume_payloads: list[tuple[str, bytes]] = []
    for rf in resumes:
        content = await rf.read()
        fname = rf.filename or "resume.bin"
        resume_payloads.append((fname, content))

    logger.info(
        "Batch analyze: %d resumes for %s (workers=%d)",
        len(resume_payloads), recruiter.company, MAX_BATCH_WORKERS,
    )

    # ── Concurrent analysis in thread pool ────────────────────────────────
    loop = asyncio.get_event_loop()
    executor = _get_executor()

    tasks = [
        loop.run_in_executor(executor, _analyze_one, fname, content, job)
        for fname, content in resume_payloads
    ]
    analyzed: list[tuple[str, AnalyzeResponse]] = list(await asyncio.gather(*tasks))

    # ── Rank by score descending ──────────────────────────────────────────
    analyzed.sort(key=lambda t: int(getattr(t[1], "score", 0) or 0), reverse=True)

    results: List[RankedResumeResult] = [
        RankedResumeResult(
            filename=fname,
            rank=idx,
            score=int(analysis.score or 0),
            ranking_reason=_ranking_reason(analysis),
            analysis=analysis,
        )
        for idx, (fname, analysis) in enumerate(analyzed, start=1)
    ]

    return BatchAnalyzeResponse(
        status="success",
        message=f"Analyzed {len(results)} resume(s) for {recruiter.company}.",
        total=len(results),
        results=results,
    )
