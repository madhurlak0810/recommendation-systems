FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*
ENV CFLAGS="-march=armv8-a"

COPY requirements.txt .
COPY temp.csv .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
