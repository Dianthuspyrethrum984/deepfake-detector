import numpy as np
from .face_extractor import FaceExtractor
from .model import DeepfakeDetector

class VideoAnalyzer:
    def __init__(self, detector, extractor=None):
        self.detector = detector
        self.extractor = extractor or FaceExtractor()

    def analyze(self, video_path, sample_rate=10):
        faces = self.extractor.extract_from_video(video_path, sample_rate)
        if not faces: return {"error": "No faces detected"}
        preds = [self.detector.predict_single(f["image"]) for f in faces]
        probs = [p["fake_probability"] for p in preds]
        avg = np.mean(probs)
        return {"verdict": "FAKE" if avg > 0.5 else "REAL", "avg_fake_probability": float(avg),
                "fake_frame_ratio": sum(1 for p in probs if p > 0.5) / len(probs), "frames_analyzed": len(preds)}
