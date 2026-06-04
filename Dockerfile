FROM python:3.12-slim

# Prevent Python from buffering stdout/stderr and avoid writing .pyc files
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --retries 5 --timeout 120 --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .
RUN mkdir -p uploads static/gradcam

EXPOSE 5005
CMD ["gunicorn", "--bind", "0.0.0.0:5005", "app:app"]
