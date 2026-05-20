from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import resume
from app.api.endpoints import recruiter

#Creates the backend application instance.
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI Powered Resume Analyzer Backend"
)

# CORS — origins are driven by settings.ALLOWED_ORIGINS (env-configurable for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(resume.router, prefix="/api/v1/resume", tags=["resume"])
app.include_router(recruiter.router, prefix="/api/v1/recruiter", tags=["recruiter"])

@app.get("/")
def root():
    return {"message": "Welcome to InSightATS API", "version": settings.VERSION}


@app.get("/health")
def health():
    """Lightweight liveness probe — Nginx / Oracle LB hits this."""
    return {"status": "ok", "version": settings.VERSION}


@app.get("/ready")
def ready():
    """Readiness probe — confirms ML models are loaded before accepting traffic."""
    from app.services.pipeline.orchestrator import _orchestrator
    loaded = _orchestrator is not None
    return {"status": "ready" if loaded else "loading", "models_loaded": loaded}
