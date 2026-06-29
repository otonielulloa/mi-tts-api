FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 💡 DIFERENTE: Descargamos los modelos aquí para que se queden grabados permanentemente en la imagen
RUN print "Descargando modelos de Kokoro..." && \
    curl -L -o kokoro-v0.19.onnx https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0.19.onnx && \
    curl -L -o voices.bin https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices.bin

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
