"""FastAPI server for deepfake detection."""

import io
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from src.face_extractor import FaceExtractor
from src.model import DeepfakeDetector
from src.preprocessor import Preprocessor
from src.video_analyzer import VideoAnalyzer

app = FastAPI(
    title="Deepfake Detection API",
    description="CNN-based deepfake detection with Grad-CAM explainability",
    version="1.0.0"
)

# Global instances (initialized on startup)
detector: DeepfakeDetector = None
face_extractor: FaceExtractor = None
preprocessor: Preprocessor = None
video_analyzer: VideoAnalyzer = None


@app.on_event("startup")
async def startup():
    """Initialize models on server startup."""
    global detector, face_extractor, preprocessor, video_analyzer

    detector = DeepfakeDetector()
    detector.build_model()
    face_extractor = FaceExtractor()
    preprocessor = Preprocessor()
    video_analyzer = VideoAnalyzer(detector, face_extractor, preprocessor)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": detector is not None and detector.model is not None
    }


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze a single image for deepfake content.

    Returns real/fake probability and optional Grad-CAM heatmap info.
    """
    import cv2

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    faces = face_extractor.extract_faces(image)

    if not faces:
        return JSONResponse(content={
            "filename": file.filename,
            "faces_detected": 0,
            "results": [],
            "message": "No faces detected in the image."
        })

    results = []
    for face_info in faces:
        preprocessed = preprocessor.preprocess(face_info["face"])
        prediction = detector.predict(preprocessed)

        heatmap = detector.grad_cam(preprocessed)

        results.append({
            "box": list(face_info["box"]),
            "fake_probability": prediction["probabilities"][0],
            "label": prediction["labels"][0],
            "heatmap_shape": list(heatmap.shape)
        })

    overall_fake = max(r["fake_probability"] for r in results)

    return JSONResponse(content={
        "filename": file.filename,
        "faces_detected": len(faces),
        "overall_fake_probability": round(overall_fake, 4),
        "overall_label": "fake" if overall_fake >= detector.threshold else "real",
        "results": results
    })


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze a video for deepfake content.

    Returns frame-by-frame analysis with aggregated confidence.
    """
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        analysis = video_analyzer.analyze_video(tmp_path)
        report = {k: v for k, v in analysis.items() if k != "frame_results"}
        report["frames_summary"] = [
            {
                "frame_index": fr["frame_index"],
                "timestamp": fr["timestamp"],
                "faces_detected": fr["faces_detected"],
                "predictions": fr["face_predictions"]
            }
            for fr in analysis.get("frame_results", [])
        ]
        return JSONResponse(content=report)
    finally:
        import os
        os.unlink(tmp_path)
