import cv2
import numpy as np
from mtcnn import MTCNN

class FaceExtractor:
    def __init__(self, min_face_size=60, confidence=0.9):
        self.detector = MTCNN(min_face_size=min_face_size)
        self.confidence = confidence

    def extract_faces(self, image, target_size=(224, 224)):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb)
        faces = []
        for det in detections:
            if det["confidence"] < self.confidence:
                continue
            x, y, w, h = det["box"]
            margin = int(max(w, h) * 0.2)
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2 = min(rgb.shape[1], x + w + margin)
            y2 = min(rgb.shape[0], y + h + margin)
            face_img = cv2.resize(rgb[y1:y2, x1:x2], target_size)
            faces.append({"image": face_img, "bbox": (x1, y1, x2, y2), "confidence": det["confidence"]})
        return faces

    def extract_from_video(self, video_path, sample_rate=10):
        cap = cv2.VideoCapture(video_path)
        results, idx = [], 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if idx % sample_rate == 0:
                for face in self.extract_faces(frame):
                    face["frame_idx"] = idx
                    results.append(face)
            idx += 1
        cap.release()
        return results
