FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Pre-compile Python bytecode to speed up cold-start module loading
RUN python -m compileall -q /app

# Run Uvicorn. The PORT environment variable will be injected by Cloud Run.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}
