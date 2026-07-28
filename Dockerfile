FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run Uvicorn. The PORT environment variable will be injected by Cloud Run.
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8001}
