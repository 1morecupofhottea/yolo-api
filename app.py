import io
import os
import base64
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image

# Setup FastAPI
app = FastAPI(title="YOLOv8 Inference API")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "yolov8_model_epoch100_batch32_lr0_0005.pt"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found at {model_path}")

model = YOLO(model_path)
model.to(device)

print(f"🚀 Model loaded successfully on {device}!")
print(f"📝 Original class names: {model.names}")

# Helper: Convert PIL Image to base64
def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

# Inference endpoint
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.4),
    iou: float = Form(0.5)
):
    try:
        # Read uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run YOLO inference
        results = model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False
        )

        detections = []
        for r in results:
            # Modify the names dictionary in the results object
            modified_names = {}
            for k, v in r.names.items():
                if v == "KhST":
                    modified_names[k] = "Aksor"
                else:
                    modified_names[k] = v
            
            # Override the names in the results object
            r.names = modified_names
            
            boxes = r.boxes
            for i, box in enumerate(boxes):
                cls_id = int(box.cls)
                conf_score = float(box.conf)
                xyxy = box.xyxy.tolist()[0]
                detections.append({
                    "object": r.names[cls_id],
                    "confidence": round(conf_score, 3),
                    "box": xyxy
                })

            # Generate annotated image with modified names
            annotated_img = Image.fromarray(r.plot()[:, :, ::-1])
            base64_img = pil_to_base64(annotated_img)

        return JSONResponse(content={
            "status": "success",
            "annotated_image": base64_img,
            "detections": detections
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
