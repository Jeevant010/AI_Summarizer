from fastapi import FastAPI
import uvicorn
import subprocess
import sys
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textSummarizer.pipeline.prediction import PredictionPipeline

app = FastAPI(title="AI Text Summarizer", version="1.0.0")

@app.get("/", tags=["authentication"] )
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def training():
    try:
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True, text=True, timeout=7200
        )
        if result.returncode != 0:
            return Response(f"Training failed!\n{result.stderr}", status_code=500)
        return Response("Training completed successfully!")
    except subprocess.TimeoutExpired:
        return Response("Training timed out!", status_code=504)
    except Exception as e:
        return Response(f"Error Occurred! {e}", status_code=500)
    
    
@app.post("/predict")
async def predict_route(text):
    try:
        
        obj = PredictionPipeline()
        text = obj.predict(text)
        return text
    except Exception as e:
        raise e
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)