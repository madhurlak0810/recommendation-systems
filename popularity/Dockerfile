# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc

# Copy project files
COPY ./app ./app
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# Command for building docker
# docker build -t popularity-app .
# docker run -d -p 8000:8000 popularity-app
