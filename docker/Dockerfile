FROM python:3.11-slim

# Avoid Python buffering issues
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only requirements first (layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ app/
COPY data/ data/

WORKDIR /app/app

CMD ["python", "train.py"]