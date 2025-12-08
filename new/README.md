# Sidewalk Bump Detection System

A computer vision system for detecting road surface irregularities (bumps, potholes, cracks) from video footage. Designed for improving safety of cyclists and scooter riders.

## Overview

This system uses a multi-stage pipeline:
1. **Audio-based Ground Truth**: Extracts bump timestamps from audio energy in specific frequency bands
2. **Video Processing**: Downscales video with edge preservation
3. **Edge Detection**: Temporal filtering of Canny edges for horizontal road features
4. **Deep Learning**: 3D CNN for temporal bump prediction

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Full Pipeline
```bash
python main.py --video data/your_video.mp4 --model simple --epochs 50
```

### Individual Components

Generate ground truth from audio:
```bash
python audio_ground_truth.py
```

Scale video:
```bash
python video_scaler.py
```

Process edges:
```bash
python edge_processor.py
```

Train model:
```bash
python train.py --model simple --epochs 50
```

Evaluate:
```bash
python evaluate.py --video data/video.mp4 --model models/simple_best.pth
```

## Architecture

### Ground Truth Generation
- Extracts audio from video using moviepy
- Applies bandpass filter (100-2000 Hz) to isolate bump frequencies
- Computes energy envelope and thresholds at 85th percentile
- Outputs timestamped bump frames with confidence scores

### Video Scaling (240p @ 15fps)
1. Convert to linear RGB space
2. Apply mild Gaussian lowpass filter
3. Lanczos downsampling
4. Unsharp masking for edge enhancement
5. Convert back to sRGB

### Edge Processing
1. Canny edge detection
2. 3D morphological opening (temporal continuity)
3. Connected component filtering (min 3 frames)
4. Orientation filtering (within 70° of horizontal)

### Model Options
- **simple**: Fast 3D CNN with 4 conv blocks
- **unet**: U-Net style encoder-decoder with skip connections
- **attention**: 2D CNN per frame with temporal attention

Input: (B, 4, 15, 240, 320) - RGB + edge channel
Output: (B, 1) - bump probability

### Training Data
- Positive: 15-frame clips ending at detected bumps
- Negative: clips with no bumps within 30 frames
- Augmentation: horizontal flip, brightness adjustment

## Configuration

Edit `config.py` for parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| TARGET_WIDTH | 320 | Output video width |
| TARGET_HEIGHT | 240 | Output video height |
| TARGET_FPS | 15 | Output framerate |
| CLIP_LENGTH | 15 | Frames per training clip |
| BATCH_SIZE | 8 | Training batch size |
| NUM_EPOCHS | 50 | Training epochs |

## Output Files

```
output/
├── ground_truth.npy      # Audio-based bump annotations
├── ground_truth_viz.png  # Visualization of detections
├── scaled_video.mp4      # Processed video
├── feature_map.mp4       # Filtered edge features
├── edge_overlay.mp4      # Edges on original video
├── training_samples.png  # Sample training clips
├── detection_results.png # Final detection plot
└── annotated_video.mp4   # Video with bump overlay

models/
└── simple_best.pth       # Best trained model
```

## Project Structure

```
new/
├── config.py             # Configuration constants
├── audio_ground_truth.py # Audio processing for labels
├── video_scaler.py       # Video downscaling
├── edge_processor.py     # Edge detection pipeline
├── data_generator.py     # Training data creation
├── model.py              # Neural network architectures
├── train.py              # Training loop
├── evaluate.py           # Evaluation and visualization
├── main.py               # Main pipeline script
└── requirements.txt      # Python dependencies
```

## License

MIT

