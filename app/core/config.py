import os

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str, default: list) -> list:
    """Read a comma-separated env var into a list, stripping whitespace."""
    v = os.getenv(name)
    if v is None:
        return default
    return [item.strip() for item in v.split(",") if item.strip()]


class Settings:
    PROJECT_NAME: str = "InSightATS API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Lightweight local storage (recruiter accounts + sessions)
    # On Oracle Cloud: set DB_PATH=/data/db.sqlite3 (persistent block volume)
    DB_PATH: str = os.environ.get("DB_PATH", os.path.join(_BASE, "db.sqlite3"))

    # Fine-tuned weights (see backend/models/)
    # On Oracle Cloud: set MODEL_DIR=/data/models  (persistent block volume)
    _model_base: str = os.environ.get("MODEL_DIR", os.path.join(_BASE, "models"))
    NER_MODEL_DIR: str        = os.path.join(os.environ.get("MODEL_DIR", os.path.join(_BASE, "models")), "ner_model")
    MATCHER_MODEL_DIR: str    = os.path.join(os.environ.get("MODEL_DIR", os.path.join(_BASE, "models")), "matcher_model")
    COMPLEXITY_MODEL_DIR: str = os.path.join(os.environ.get("MODEL_DIR", os.path.join(_BASE, "models")), "complexity_model")

    # SHAP explainability is very slow (~60s+ per request); off by default
    ENABLE_SHAP: bool = _env_bool("ENABLE_SHAP", False)

    # CORS — set ALLOWED_ORIGINS as a comma-separated list in the environment.
    # Example: ALLOWED_ORIGINS=https://insightats.vercel.app,https://www.insightats.com
    ALLOWED_ORIGINS: list = _env_list(
        "ALLOWED_ORIGINS",
        default=[
            "http://localhost",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
    )

    # Clerk — required for POST /api/v1/recruiter/batch-analyze (frontend getToken() JWTs)
    # Issuer URL from Clerk Dashboard → API Keys → Advanced → JWT issuer
    CLERK_ISSUER: str = os.environ.get("CLERK_ISSUER", "").rstrip("/")
    CLERK_JWKS_URL: str = os.environ.get(
        "CLERK_JWKS_URL",
        f"{CLERK_ISSUER}/.well-known/jwks.json" if CLERK_ISSUER else "",
    )
    # Optional: Clerk "Authorized parties" / azp — leave empty for default session tokens
    CLERK_AUDIENCE: str = os.environ.get("CLERK_AUDIENCE", "").strip()

    # Redis caching — set REDIS_URL to enable result caching (e.g. redis://localhost:6379/0)
    # Leave empty to disable Redis and run without caching (graceful degradation).
    REDIS_URL: str = os.environ.get("REDIS_URL", "").strip()
    # How long to keep a cached analysis result (seconds). Default: 1 hour.
    CACHE_TTL: int = int(os.environ.get("CACHE_TTL", "3600"))


settings = Settings()
