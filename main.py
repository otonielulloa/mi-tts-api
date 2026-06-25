from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import edge_tts
import os

app = FastAPI()

class TTSRequest(BaseModel):
    input: str
    voice: str = "es-MX-JorgeNeural"  # Voz ultra-natural masculina. 
    # Otras excelentes opciones en español: 
    # "es-ES-AlvaroNeural" (España), "es-MX-DaliaNeural" (Mujer México), "es-US-AlonsoNeural" (EEUU)

@app.post("/v1/audio/speech")
async def text_to_speech(request: TTSRequest):
    if not request.input:
        raise HTTPException(status_code=400, detail="Falta el texto de entrada (input)")
    
    output_filename = f"audio-{os.getpid()}.mp3"
    
    try:
        # Generar la voz usando la red neuronal de Microsoft
        communicate = edge_tts.Communicate(request.input, request.voice)
        await communicate.save(output_filename)
        
        # Enviar el archivo de vuelta a n8n y borrarlo después para no llenar el disco
        return FileResponse(
            output_filename, 
            media_type="audio/mpeg", 
            filename="voice.mp3"
        )
    except Exception as e:
        if os.path.exists(output_filename):
            os.remove(output_filename)
        raise HTTPException(status_code=500, detail=str(e))
