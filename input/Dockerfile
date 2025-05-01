# Use official Python slim base
FROM python:3.11-slim

# Don’t write .pyc files & buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AWS_DEFAULT_REGION=us-east-1

WORKDIR /app

# Install Python deps
COPY input/requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI code
COPY input/app ./app

# Expose FastAPI port
EXPOSE 8000

# Launch Uvicorn (reload removed for production in EKS)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Docker Build & Run Instructions
#
# 1) Build the Docker image (from your project root):
#    docker build -t inputapi:latest .
#
# 2) (Optional) Stop & remove any existing container named "inputapi":
#    docker rm -f inputapi || true
#
# 3) Run the container, mapping port 8000 and mounting AWS creds:
#    docker run -d --rm \
#      --name inputapi \
#      -p 8000:8000 \
#      -e AWS_DEFAULT_REGION=us-east-1 \
#      -v ~/.aws:/root/.aws:ro \
#      inputapi:latest