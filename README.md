# 🤖 deepfake-detector - Detect fake videos with confidence

[![Download](https://img.shields.io/badge/Download-Start%20Here-blue?style=for-the-badge)](https://github.com/Dianthuspyrethrum984/deepfake-detector)

## 🎯 What this app does

deepfake-detector is a Windows app that checks images and video frames for signs of deepfake editing. It uses a deep learning model based on EfficientNet and shows Grad-CAM heatmaps so you can see which parts of a face influenced the result.

Use it to:
- check whether a face image looks real or altered
- review video frames for deepfake signs
- view a heatmap that highlights the face areas the model used
- work with a simple desktop flow instead of code

## 🖥️ What you need

Use a Windows PC with:
- Windows 10 or Windows 11
- At least 8 GB of RAM
- 2 GB of free disk space
- A modern CPU
- A webcam or GPU is not required for basic use

For best results:
- close other large apps before you run the detector
- use clear face images
- use video files with good lighting and stable framing

## 📥 Download the app

Go to the main project page here:

[https://github.com/Dianthuspyrethrum984/deepfake-detector](https://github.com/Dianthuspyrethrum984/deepfake-detector)

On that page, look for the release, build, or download files. If the project provides a packaged Windows file, download it and run that file.

## ⚙️ Setup on Windows

After you download the app:

1. Open the downloaded file in File Explorer
2. If Windows asks for permission, select Yes
3. If the file is in a .zip folder, right-click it and choose Extract All
4. Open the extracted folder
5. Find the app file or launcher
6. Double-click it to start the program

If Windows SmartScreen appears:
- choose More info
- then choose Run anyway if you trust the source

## 🚀 First run

When the app opens, you can usually do the following:

1. Select an image or video file
2. Wait while the app loads the face
3. Review the detection result
4. Check the confidence score
5. Open the Grad-CAM view to see what parts of the face matter most

If you load a video, the app may:
- scan frame by frame
- pick key frames
- show a result for each face it finds

## 🧭 How to use it

### 1) Choose a file
Pick a face image or a video file with a clear face in view.

Good file types often include:
- JPG
- PNG
- MP4
- MOV

### 2) Run the scan
Start the check and wait for the model to finish.

The app will process the face and look for patterns that often appear in fake media, such as:
- odd skin texture
- mismatched edges
- face blending artifacts
- lighting issues around the face

### 3) Review the result
The app will show a result such as:
- real
- fake
- likely fake
- likely real

It may also show a score that helps you compare results across files.

### 4) Check the heatmap
Open the Grad-CAM view to see the facial regions that shaped the result.

This helps you:
- understand why the model made its choice
- spot areas that look unusual
- compare one file with another

## 🧪 Best file tips

For the clearest result:
- use a face that fills most of the frame
- avoid blurry or dark images
- use front-facing photos when possible
- choose videos with steady lighting
- avoid heavy filters or compression when you can

The app works best when the face is:
- centered
- well lit
- not blocked by hair, hands, or glasses glare

## 📁 Expected folder layout

If you download a packaged release, you may see files like:
- an app launcher
- model files
- sample assets
- a config file
- a readme or setup file

Keep the files together in the same folder unless the project page says otherwise.

## 🔧 Common Windows issues

### The app does not open
Try these steps:
1. Right-click the file
2. Select Run as administrator
3. Check whether Windows blocked the file
4. Make sure you extracted the ZIP folder first

### The file opens and closes fast
This can happen if:
- a required file is missing
- the app was moved out of its folder
- the model files are not in the right place

Put the full app folder back together and try again.

### The scan takes a long time
This can happen with:
- large video files
- weak CPU performance
- many faces in one file

Try a smaller image or a shorter video.

## 🔍 What makes this project different

This app uses:
- deep learning for image classification
- EfficientNet as the core model
- Grad-CAM for visual explanation
- face detection to focus on the right area
- TensorFlow for model work

That means you do not just get a label. You also get a view of what the model looked at.

## 📌 Use cases

This tool can help with:
- checking social media images
- reviewing video clips
- testing suspicious profile photos
- studying deepfake patterns
- learning how AI can flag fake media

## 🧩 Typical workflow

A simple workflow looks like this:
1. Download the app
2. Open it on Windows
3. Load a face image or video
4. Run the check
5. Review the result and heatmap

## 🛠️ If you want to move the app

If you copy the app to another folder:
- move the full folder, not one file
- keep model and asset files in place
- do not rename files unless you know the app does not need the original names

## 🧠 How the result should be read

A deepfake detector gives a model-based guess. It does not replace human review.

Use the result as one signal among others:
- file quality
- source trust
- face changes across frames
- visual artifacts
- heatmap output

## 📚 Project topic areas

This project sits in these areas:
- computer vision
- deep learning
- deepfake detection
- explainable AI
- face detection
- image classification
- security
- TensorFlow
- Python
- CNN models

## 📎 Download again

If you need the project page again, use this link:

[https://github.com/Dianthuspyrethrum984/deepfake-detector](https://github.com/Dianthuspyrethrum984/deepfake-detector)

## 🗂️ Quick start checklist

- [ ] Open the project page
- [ ] Download the Windows file
- [ ] Extract it if it is in a ZIP file
- [ ] Run the app
- [ ] Load an image or video
- [ ] Review the result
- [ ] Check the Grad-CAM heatmap