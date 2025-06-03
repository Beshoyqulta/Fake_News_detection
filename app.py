
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from functions import generate_content, NewsRequest

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rashed-five.vercel.app",  # Your frontend
        "https://rashed-five.vercel.app/",  # Handle trailing slash
        "http://localhost:3000",  # Local dev (Next.js)
        "http://localhost:5173",  # Local dev (Vite)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.post("/Trustness/")
async def translate(request: NewsRequest):
    try:
        trusted_text = generate_content(request.input_str)
        return {"trusted_text": trusted_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
    