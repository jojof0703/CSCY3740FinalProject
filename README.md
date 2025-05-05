# DeepFake Detection System

A deep learning-based system for detecting deepfake videos using EfficientNetB0 and custom feature extraction.

## Features

- Video frame extraction and preprocessing
- Custom feature extraction for deepfake detection
- EfficientNetB0-based deep learning model
- Real-time video analysis with face detection
- Dataset analysis and visualization tools

## Project Structure

```
project_root/
├── videos/          # Input videos directory
├── dataset/         # Extracted frames
├── split/          # Train/val split
│   ├── train/
│   └── val/
├── checkpoints/    # Model weights
└── output/         # Analysis results
```

## Setup

1. Create the required directories:
```bash
mkdir videos dataset split checkpoints output
```

2. Install dependencies:
```bash
pip install tensorflow opencv-python numpy tqdm face-recognition termcolor ascii-magic
```

## Usage

1. Extract frames from videos:
```bash
python extract_frames.py
```

2. Split dataset into train/val:
```bash
python split_dataset.py
```

3. Train the model:
```bash
python train_efficientnet.py
```

4. Test the model:
```bash
python test_model.py
```

5. Analyze a video:
```bash
python detect_deepfake_video.py
```

6. Analyze dataset statistics:
```bash
python analyze_dataset.py
```

## Model Architecture

The system uses a hybrid approach combining:
- EfficientNetB0 for image feature extraction
- Custom feature extraction for video-specific metrics
- Attention mechanism for improved detection
- Face detection for targeted analysis

## License

This project is open source and available under the MIT License. 