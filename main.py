import argparse, cv2
from src.model import DeepfakeDetector
from src.face_extractor import FaceExtractor
from src.video_analyzer import VideoAnalyzer

def main():
    p = argparse.ArgumentParser(description="Deepfake Detection System")
    p.add_argument("--mode", choices=["predict", "analyze-video", "api"], default="predict")
    p.add_argument("--input", help="Input path")
    p.add_argument("--model-path", default="models/deepfake_detector.keras")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    det = DeepfakeDetector()
    if args.mode == "predict":
        det.load(args.model_path)
        for i, f in enumerate(FaceExtractor().extract_faces(cv2.imread(args.input))):
            r = det.predict_single(f["image"])
            print(f"Face {i+1}: {r['label']} (prob={r['fake_probability']:.3f})")
    elif args.mode == "analyze-video":
        det.load(args.model_path)
        r = VideoAnalyzer(det).analyze(args.input)
        print(f"Verdict: {r['verdict']} (avg={r['avg_fake_probability']:.3f})")
    elif args.mode == "api":
        import uvicorn; uvicorn.run("src.api:app", host="0.0.0.0", port=args.port)

if __name__ == "__main__": main()
