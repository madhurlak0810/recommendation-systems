# Use official Python lightweight image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory inside the container
WORKDIR /app

# Copy app code
COPY ./app ./app

# Copy requirements.txt if you have one (optional)
# COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn requests

# Expose FastAPI default port
EXPOSE 8000

# Command to run FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
