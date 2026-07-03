import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.analyze import AnalyzeResponse
from app.schemas.job import JobDescription
from app.services.pipeline.orchestrator import get_orchestrator
from app.services.resume_parser import ResumeParser

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    resume_file: UploadFile = File(...),
    job_description: str = Form(...),
):
    """
    InSightATS analyze — multipart FormData:
    - `resume_file`: PDF, DOCX, or TXT
    - `job_description`: JSON string ``{ "title", "description", "mandatory_skills"?, "preferred_skills"? }``

    Results are cached in Redis (if configured) keyed by sha256(resume + jd)
    so identical pairs return in <5 ms on subsequent requests.
    """
    try:
        content = await resume_file.read()
        filename = resume_file.filename or "resume.bin"

        text = ResumeParser.extract_text(content, filename)

        try:
            job_dict = json.loads(job_description)
            job = JobDescription(**job_dict)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid job_description JSON: {e}") from e

        # Use the cache-aware async entry point.
        result = await get_orchestrator().analyze_cached(text, job)

        return result

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
