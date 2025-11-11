import io
import os
import base64
<<<<<<< HEAD
import logging
from typing import Optional
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZIPMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
from functools import lru_cache
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup FastAPI with optimizations
app = FastAPI(
    title="YOLOv8 Inference API",
    description="Optimized YOLO object detection API for Render",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add GZIP compression middleware
app.add_middleware(GZIPMiddleware, minimum_size=1000)

# CORS middleware
=======
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image

# Setup FastAPI
app = FastAPI(title="YOLOv8 Inference API")

# Allow frontend requests
>>>>>>> 190eb4367bfd6ae377673866d1ef53926625e222
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< HEAD
# Thread pool for CPU-bound tasks (optimized for Render's CPU limits)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Model configuration
device = "cpu"  # Render free tier doesn't have GPU
model_path = os.getenv("MODEL_PATH", "yolov8_model_epoch100_batch32_lr0_0005.pt")

# Global model variable
model: Optional[YOLO] = None

def load_model():
    """Load YOLO model with error handling"""
    global model
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found at {model_path}")
        
        logger.info(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        model.to(device)
        
        # Optimize for CPU
        torch.set_num_threads(MAX_WORKERS)
        
        logger.info(f"✅ Model loaded successfully on {device}")
        logger.info(f"📝 Class names: {model.names}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("🚀 Starting API server...")
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")
    logger.info(f"Worker threads: {MAX_WORKERS}")

# Cache for class name mapping
@lru_cache(maxsize=1)
def get_modified_names():
    """Cache the modified names dictionary"""
    if model is None:
        return {}
    modified_names = {}
    for k, v in model.names.items():
        modified_names[k] = "Aksor" if v == "KhST" else v
    return modified_names

# Helper: Convert PIL Image to base64 (optimized)
def pil_to_base64(img: Image.Image, quality: int = 85) -> str:
    """Convert PIL Image to base64 with optimized quality"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

# Async image preprocessing with size limits
async def preprocess_image(image_bytes: bytes, max_size: int = 1280) -> Image.Image:
    """Load and preprocess image asynchronously with size optimization"""
    loop = asyncio.get_event_loop()
    
    def _load_and_resize():
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Resize large images to reduce processing time
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized image to {new_size}")
        return img
    
    image = await loop.run_in_executor(executor, _load_and_resize)
    return image

# Run inference in thread pool
def run_inference(image: Image.Image, conf: float, iou: float):
    """Run YOLO inference synchronously in thread pool"""
    if model is None:
        raise RuntimeError("Model not loaded")
    
    with torch.no_grad():
        results = model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
            imgsz=640  # Fixed size for consistent performance
        )
    return results

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "🚀 YOLOv8 Inference API",
        "status": "running",
        "device": device,
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint for Render"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "device": device,
        "model_loaded": model is not None,
        "workers": MAX_WORKERS
    }
=======
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

@app.get("/")
async def root():
    return {"message": "🚀 YOLOv8 Inference API is running!"}
>>>>>>> 190eb4367bfd6ae377673866d1ef53926625e222

# Inference endpoint
@app.post("/predict")
async def predict(
<<<<<<< HEAD
    file: UploadFile = File(..., description="Image file to analyze"),
    conf: float = Form(0.4, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Form(0.5, ge=0.0, le=1.0, description="IoU threshold"),
    image_quality: int = Form(85, ge=50, le=100, description="Output image quality")
):
    """
    Object detection endpoint
    - Accepts image files
    - Returns annotated image and detections
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read uploaded image
        logger.info(f"Processing image: {file.filename}")
        image_bytes = await file.read()
        
        # Check file size (max 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Preprocess image
        image = await preprocess_image(image_bytes)
        
        # Run inference
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            run_inference,
            image,
            conf,
            iou
        )
        
        # Get cached modified names
        modified_names = get_modified_names()
        
        detections = []
        annotated_img = None
        
        for r in results:
            r.names = modified_names
            boxes = r.boxes
            
            # Process detections
            detections = [
                {
                    "object": r.names[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                    "box": [round(coord, 2) for coord in box.xyxy.tolist()[0]]
                }
                for box in boxes
            ]
            
            # Generate annotated image
            plot_result = await loop.run_in_executor(executor, r.plot)
            annotated_img = Image.fromarray(plot_result[:, :, ::-1])
        
        # Convert to base64
        base64_img = await loop.run_in_executor(
            executor,
            pil_to_base64,
            annotated_img,
            image_quality
        )
        
        # Clean up
        del image, results, annotated_img
        gc.collect()
        
        logger.info(f"✅ Detected {len(detections)} objects")
        
        return JSONResponse(content={
            "status": "success",
            "annotated_image": base64_img,
            "detections": detections,
            "count": len(detections)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down...")
    executor.shutdown(wait=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("✅ Cleanup complete")
=======
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
>>>>>>> 190eb4367bfd6ae377673866d1ef53926625e222
