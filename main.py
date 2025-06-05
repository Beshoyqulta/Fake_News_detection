from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from functions import generate_content, NewsRequest

app = FastAPI(title="News Trustiness API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rashed-five.vercel.app",
        "https://rashed-five.vercel.app/",
        "http://localhost:3000",
        "http://localhost:5173",
        "*"  # Allow all origins for production (adjust as needed)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.get("/")
async def root():
    return {"message": "News Trustiness API is running!"}

@app.post("/Trustness/")
async def analyze_trustiness(request: NewsRequest):
    try:
        trusted_text = generate_content(request.input_str)
        return {"trusted_text": trusted_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
