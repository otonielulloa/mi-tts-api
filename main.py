import asyncio
import os
import time
import re
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro  # 💡 Corregido: Se importa 'Kokoro' en lugar de 'KokoroOnnx'

app = FastAPI()

print("Inicializando Motores de IA Locales...")

# 💡 1. Cargamos Kokoro de forma local (aprox 350MB RAM)
if os.path.exists("kokoro-v0.19.onnx") and os.path.exists("voices.bin"):
    # 💡 Corregido: Se usa la clase 'Kokoro'
    kokoro = Kokoro("kokoro-v0.19.onnx", "voices.bin")
    print("✓ Kokoro-82M cargado con éxito.")
else:
    print("✗ Error: No se encontraron los archivos de modelo de Kokoro.")
    kokoro = None

# 💡 2. Whisper en modelo 'small' con optimización para tus 2 vCPUs
model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=2)
print("✓ Whisper Small listo para escuchar.")

class TTSRequest(BaseModel):
    input: str
    voice: str = "am_adam"

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def remove_file(path: str):
    """Elimina archivos temporales tras ser descargados"""
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"Archivo temporal eliminado: {path}")
        except Exception as e:
            print(f"Error al eliminar {path}: {e}")

@app.get("/v1/audio/download/{filename}")
async def download_file(filename: str, background_tasks: BackgroundTasks):
    if os.path.exists(filename):
        background_tasks.add_task(remove_file, filename)
        return FileResponse(filename, media_type="audio/wav", filename=filename)
    raise HTTPException(status_code=404, detail="Archivo no encontrado")

@app.post("/v1/audio/generate")
async def generate_unified(request: Request, req_body: TTSRequest):
    if not req_body.input:
        raise HTTPException(status_code=400, detail="Falta el texto de entrada (input)")
    if not kokoro:
        raise HTTPException(status_code=500, detail="El motor Kokoro no está inicializado.")
    
    timestamp = int(time.time() * 1000)
    audio_filename = f"voice-{timestamp}.wav"
    
    try:
        # 💡 Segmentación inteligente por puntuación para evitar saturación de CPU y Timeout 502
        frases = [f.strip() for f in re.split(r'[.!?\n]+', req_body.input) if f.strip()]
        
        if not frases:
            raise HTTPException(status_code=400, detail="El texto provisto no contiene frases válidas.")
        
        audio_fragmentos = []
        sample_rate = 24000  # Frecuencia nativa de Kokoro
        
        for frase in frases:
            # Procesamos cada frase secuencialmente de forma asíncrona en el hilo secundario
            samples, rate = await asyncio.to_thread(
                kokoro.create, frase, voice=req_body.voice, speed=1.0
            )
            sample_rate = rate
            audio_fragmentos.append(samples)
            
        # Unimos todos los fragmentos matemáticamente en un solo bloque de audio
        samples_totales = np.concatenate(audio_fragmentos)
        
        # Guardamos el archivo final en el disco NVMe
        await asyncio.to_thread(sf.write, audio_filename, samples_totales, sample_rate)
        
        # Transcripción asíncrona usando Whisper Small
        segments, info = await asyncio.to_thread(
            model.transcribe, audio_filename, language="es", word_timestamps=True
        )
        
        words = []
        for segment in segments:
            if segment.words:
                for w in segment.words:
                    words.append({
                        "text": w.word.strip(),
                        "start": w.start,
                        "end": w.end
                    })
        
        vtt_lines = ["WEBVTT\n"]
        if words:
            chunk_size = 3
            for i in range(0, len(words), chunk_size):
                chunk = words[i:i+chunk_size]
                for idx in range(len(chunk)):
                    start_time = format_time(chunk[idx]["start"])
                    if idx < len(chunk) - 1:
                        end_time = format_time(chunk[idx+1]["start"])
                    else:
                        end_time = format_time(chunk[idx]["end"])
                    
                    processed_words = []
                    for j, w in enumerate(chunk):
                        word_text = w["text"].upper()
                        if j == idx:
                            processed_words.append(f'<font color="#FFFF00">{word_text}</font>')
                        else:
                            processed_words.append(word_text)
                    
                    phrase = " ".join(processed_words)
                    vtt_lines.append(f"{start_time} --> {end_time}\n{phrase}\n")
                    
            total_duration = words[-1]["end"]
        else:
            raise Exception("Whisper no detectó palabras en el audio unificado.")

        subtitles_text = "\n".join(vtt_lines)
        base_url = str(request.base_url)
        audio_url = f"{base_url}v1/audio/download/{audio_filename}"
        
        return JSONResponse(content={
            "audio_url": audio_url,
            "subtitles": subtitles_text,
            "duration": total_duration
        })
        
    except Exception as e:
        if os.path.exists(audio_filename):
            os.remove(audio_filename)
        raise HTTPException(status_code=500, detail=str(e))
