import asyncio
import os
import time
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
import edge_tts
from faster_whisper import WhisperModel

app = FastAPI()

print("Cargando modelo Faster-Whisper Medium...")
# 💡 Configurado optimizadamente para los 2 vCPU cores de tu servidor KVM
model = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=2)
print("Whisper Medium listo para escuchar.")

class TTSRequest(BaseModel):
    input: str
    voice: str = "es-MX-JorgeNeural"

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def remove_file(path: str):
    """Elimina archivos temporales de forma segura tras ser descargados"""
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"Archivo temporal eliminado con éxito: {path}")
        except Exception as e:
            print(f"Error al eliminar el archivo temporal {path}: {e}")

@app.get("/v1/audio/download/{filename}")
async def download_file(filename: str, background_tasks: BackgroundTasks):
    if os.path.exists(filename):
        # 💡 Tarea en segundo plano: se borra el archivo justo después de enviarlo al usuario
        background_tasks.add_task(remove_file, filename)
        return FileResponse(filename, media_type="audio/mpeg", filename=filename)
    raise HTTPException(status_code=404, detail="Archivo no encontrado")

@app.post("/v1/audio/generate")
async def generate_unified(request: Request, req_body: TTSRequest):
    if not req_body.input:
        raise HTTPException(status_code=400, detail="Falta el texto de entrada (input)")
    
    timestamp = int(time.time() * 1000)
    audio_filename = f"voice-{timestamp}.mp3"
    
    try:
        communicate = edge_tts.Communicate(req_body.input, req_body.voice)
        await communicate.save(audio_filename)
        
        # 💡 Transcripción asíncrona usando el modelo Medium en un hilo secundario
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
            raise Exception("Whisper no pudo detectar palabras en el audio.")

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

@app.post("/v1/audio/speech")
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    if not request.input:
        raise HTTPException(status_code=400, detail="Falta el texto de entrada (input)")
    output_filename = f"audio-{os.getpid()}-{int(time.time())}.mp3"
    try:
        communicate = edge_tts.Communicate(request.input, request.voice)
        await communicate.save(output_filename)
        # 💡 Limpieza automática para este endpoint también
        background_tasks.add_task(remove_file, output_filename)
        return FileResponse(output_filename, media_type="audio/mpeg", filename="voice.mp3")
    except Exception as e:
        if os.path.exists(output_filename):
            os.remove(output_filename)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/subtitles")
async def generate_subtitles(request: TTSRequest):
    if not request.input:
        raise HTTPException(status_code=400, detail="Falta el texto de entrada (input)")
    
    timestamp = int(time.time() * 1000)
    temp_audio = f"temp-{timestamp}.mp3"
    try:
        communicate = edge_tts.Communicate(request.input, request.voice)
        await communicate.save(temp_audio)
        
        # 💡 Transcripción asíncrona con Faster-Whisper Medium
        segments, info = await asyncio.to_thread(
            model.transcribe, temp_audio, language="es", word_timestamps=True
        )
        
        vtt_lines = ["WEBVTT\n"]
        words = []
        for segment in segments:
            if segment.words:
                for w in segment.words:
                    words.append({"text": w.word.strip(), "start": w.start, "end": w.end})
                
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
                    
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
            
        return PlainTextResponse("\n".join(vtt_lines), media_type="text/vtt")
    except Exception as e:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        raise HTTPException(status_code=500, detail=str(e))
