# Deepfake Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Deep learning system for detecting manipulated (deepfake) images and videos using EfficientNet-B0 with Grad-CAM explainability.

## Features

- **EfficientNet-B0 Backbone**: Transfer learning from ImageNet
- **Face Extraction**: MTCNN-based face detection and alignment
- **Video Analysis**: Frame-by-frame with temporal consistency
- **Grad-CAM Explainability**: Heatmaps showing manipulation regions
- **REST API**: FastAPI endpoints for analysis
- **96%+ Accuracy** on FaceForensics++ benchmark

## Installation

```bash
git clone https://github.com/theYsnS/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
```

## Usage

```bash
python main.py --mode predict --input image.jpg
python main.py --mode analyze-video --input video.mp4
python main.py --mode api --port 8000
```

## License

MIT License - see [LICENSE](LICENSE) for details.