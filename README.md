# YOLOv8 Inference API

Optimized FastAPI service for YOLO object detection, designed for Render deployment.

## Features

- ⚡ Async processing with ThreadPoolExecutor
- 🗜️ GZIP compression for responses
- 🎯 Automatic image resizing for performance
- 🔄 Memory management with garbage collection
- 📊 Proper logging and health checks
- 🚀 Optimized for CPU inference on Render

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check (for Render)
- `POST /predict` - Object detection
- `GET /docs` - Interactive API documentation

## Deployment on Render

1. Push your code to GitHub
2. Connect Render to your repository
3. Render will automatically detect `render.yaml`
4. Deploy!

## Environment Variables

- `MAX_WORKERS` - Number of worker threads (default: 2)
- `MODEL_PATH` - Path to YOLO model file
- `PORT` - Server port (Render sets this automatically)

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app:app --reload --port 8000
```

## API Usage

```python
import requests

# Predict
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f},
        data={'conf': 0.4, 'iou': 0.5}
    )
    
result = response.json()
print(f"Found {result['count']} objects")
```

## Performance Optimizations

- CPU-only PyTorch (smaller, faster on Render free tier)
- Image resizing to max 1280px
- JPEG quality optimization (85%)
- Thread pool for concurrent requests
- Automatic memory cleanup
- GZIP compression for responses

## Model

YOLOv8 trained for Khmer character detection (Aksor).
