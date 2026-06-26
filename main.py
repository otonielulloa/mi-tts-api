from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
import edge_tts
import os

app = FastAPI()

class TTSRequest(BaseModel):
    input: str
    voice: str = "es-MX-JorgeNeural"

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

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
    try:
        communicate = edge_tts.Communicate(request.input, request.voice)
        words = []
        async for chunk in communicate.stream():
            if chunk["type"] == "WordBoundary":
                words.append({
                    "text": chunk["text"],
                    "start": chunk["offset"] / 10000000,
                    "end": (chunk["offset"] + chunk["duration"]) / 10000000
                })
        
        vtt_lines = ["WEBVTT\n"]
        # Agrupamos de a 3 palabras para un estilo TikTok dinámico
        for i in range(0, len(words), 3):
            group = words[i:i+3]
            if not group:
                continue
            start_time = format_time(group[0]["start"])
            end_time = format_time(group[-1]["end"])
            phrase = " ".join([w["text"] for w in group]).upper()
            vtt_lines.append(f"{start_time} --> {end_time}\n{phrase}\n")
        
        return PlainTextResponse("\n".join(vtt_lines), media_type="text/vtt")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
