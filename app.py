from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Load model once when server starts
model = YOLO("best.pt")

@app.get("/")
def home():
    return {"message": "YOLOv8 API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    results = model(image)
    detections = results[0].tojson()  # JSON output
    return {"detections": detections}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
