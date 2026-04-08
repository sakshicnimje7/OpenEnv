FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TASK_DIFFICULTY=easy
ENV MODEL_NAME=gpt-3.5-turbo
ENV API_BASE_URL=
ENV HF_TOKEN=
ENV OPENAI_API_KEY=
ENV PORT=7860

# Expose service port used by HF Spaces Docker runtime
EXPOSE 7860

# Run API server by default
CMD ["python", "-m", "server.app"]
