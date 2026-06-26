from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
import edge_tts
import os
import time
import whisper

app = FastAPI()

print("Cargando modelo Whisper Pro...")
model = whisper.load_model("tiny")
print("Whisper listo para el resaltado dinámico.")

class TTSRequest(BaseModel):
    input: str
    voice: str = "es-MX-JorgeNeural"

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

@app.get("/v1/audio/download/{filename}")
async def download_file(filename: str):
    if os.path.exists(filename):
        return FileResponse(filename, media_type="audio/mpeg", filename=filename)
    raise HTTPException(status_code=404, detail="Archivo no encontrado")

def generar_vtt_resaltado(words):
    """
    💡 ALGORITMO DE RESALTADO ESTILO TIKTOK:
    Agrupa de 3 en 3 palabras, pero genera un evento por cada palabra individual
    para pintarla de amarillo brillante (#FFFF00) mientras se pronuncia.
    """
    vtt_lines = ["WEBVTT\n"]
    
    # Agrupamos los índices de 3 en 3
    for i in range(0, len(words), 3):
        group = words[i:i+3]
        if not group:
            continue
        
        # Para cada palabra dentro de este grupo de 3, creamos un bloque de tiempo exclusivo
        for target_index in range(len(group)):
            word_actual = group[target_index]
            
            # El tiempo de este cuadro de animación corresponde a la palabra activa
            start_time = format_time(word_actual["start"])
            end_time = format_time(word_actual["end"])
            
            # Construimos la frase formateando con color la palabra que está sonando
            frase_formateada = []
            for idx, w in enumerate(group):
                word_text = w["text"].upper()
                if idx == target_index:
                    # Pintamos de amarillo la palabra activa
                    frase_formateada.append(f"<font color=\"#FFFF00\">{word_text}</font>")
                else:
                    # Las palabras compañeras se quedan en blanco normal
                    frase_formateada.append(f"<font color=\"#FFFFFF\">{word_text}</font>")
            
            texto_final = " ".join(frase_formateada)
            vtt_lines.append(f"{start_time} --> {end_time}\n{texto_final}\n")
            
    return "\n".join(vtt_lines)

@app.post("/v1/audio/generate")
async def generate_unified(request: Request, req_body: TTSRequest):
    if not req_body.input:
        raise HTTPException(status_code=400, detail="Falta el texto de entrada (input)")
    
    timestamp = int(time.time() * 1000)
    audio_filename = f"voice-{timestamp}.mp3"
    
    try:
        # 1. Generar audio nativo
        communicate = edge_tts.Communicate(req_body.input, req_body.voice)
        await communicate.save(audio_filename)
        
        # 2. Transcribir con Whisper para obtener milisegundos exactos
        result = model.transcribe(audio_filename, word_timestamps=True, language="es")
        
        words = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                words.append({
                    "text": w["word"].strip(),
                    "start": w["start"],
                    "end": w["end"]
                })
        
        if words:
            subtitles_text = generar_vtt_resaltado(words)
            total_duration = words[-1]["end"]
        else:
            raise Exception("Whisper no detectó fonemas en el audio.")

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
async def text_to_speech(request: TTSRequest):
    if not request.input:
        raise HTTPException(status_code=400, detail="Falta el texto de entrada (input)")
    output_filename = f"audio-{os.getpid()}.mp3"
    try:
        communicate = edge_tts.Communicate(request.input, request.voice)
        await communicate.save(output_filename)
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
        
        result = model.transcribe(temp_audio, word_timestamps=True, language="es")
        words = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                words.append({
                    "text": w["word"].strip(),
                    "start": w["start"],
                    "end": w["end"]
                })
        
        subtitles_text = generar_vtt_resaltado(words)
        
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
            
        return PlainTextResponse(subtitles_text, media_type="text/vtt")
    except Exception as e:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        raise HTTPException(status_code=500, detail=str(e))
