from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from functions import generate_content, NewsRequest
app = FastAPI()
@app.post("/Trustness/")  # This line decorates 'translate' as a POST endpoint
async def translate(request: NewsRequest):
    try:
        # Call your translation function
        trusted_text = generate_content(request.input_str)
        return {"trusted_text": trusted_text}
    except Exception as e:
        # Handle exceptions or errors during translation
        raise HTTPException(status_code=500, detail=str(e))
