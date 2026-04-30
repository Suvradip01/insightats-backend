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
    - `job_description`: JSON string `{ "title", "description", "mandatory_skills"?, "preferred_skills"? }`
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

        result = get_orchestrator().analyze(text, job)

        if result.status == "pending_setup":
            return result

        return result

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
