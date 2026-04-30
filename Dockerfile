# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.11-slim

# System deps needed by pdfplumber (poppler) and python-docx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install them as the user
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install --no-cache-dir huggingface_hub

# Download the models from a dedicated HF Model Repo to bypass the 1 GB Space limit
RUN huggingface-cli download Suvradip01/insightats-models --local-dir /app/models --repo-type model

# Copy application source (models are no longer needed in the space repo)
COPY --chown=user . /app

# Hugging Face Spaces use port 7860
ENV PORT=7860
ENV APP_ENV=production
# We store the DB in the app directory. Note: On HF Spaces Free tier, this resets if the space sleeps.
ENV DB_PATH=/app/db.sqlite3
ENV MODEL_DIR=/app/models

CMD ["python", "run.py"]
