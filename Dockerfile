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

# Wake identity-service in the background while this process still imports.
CMD ["sh", "-c", "python /app/scripts/wake_identity.py & exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
