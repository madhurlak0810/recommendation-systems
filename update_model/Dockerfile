FROM python:3.11-slim

WORKDIR /app

COPY update.py .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
