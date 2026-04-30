# InSightATS Backend (FastAPI)

## Prerequisites

- Python 3.10+ (your machine: 3.12 works)
- pip

## Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Models (local-only)**

This backend can run with or without fine-tuned model weights.

- If model weights are **not present**, the API returns `status="pending_setup"` (or uses heuristics for M3).
- If you place weights under `backend/models/`, the runners will load them at request-time.

Expected folders:

- `backend/models/ner_model/` (M1: token classification NER)
- `backend/models/matcher_model/` (M2: sequence classification match)
- `backend/models/complexity_model/` (M3: project complexity; optional — has a heuristic fallback)

## Running the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.
Docs are available at `http://127.0.0.1:8000/docs`.

## API

- `POST /api/v1/resume/analyze`
  - multipart form-data:
    - `resume_file`: PDF/DOCX/TXT
    - `job_description`: JSON string: `{ "title": "...", "description": "...", "mandatory_skills"?: [...], "preferred_skills"?: [...] }`
