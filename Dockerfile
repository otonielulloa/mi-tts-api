FROM python:3.10-slim

# Instalar dependencias del sistema (incluyendo FFmpeg y librerías de aceleración de CPU)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
