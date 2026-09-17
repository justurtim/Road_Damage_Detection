# Road Damage Detection System

A computer vision system that detects **potholes** and **cracks** on road surfaces from images, video files, or a live webcam feed — built with classical image processing (OpenCV), no trained model or GPU required.

## Features

- Detects two types of road damage:
  - **Cracks** — thin, elongated dark structures
  - **Potholes** — roundish, blob-like dark regions
- Works on three input types:
  - Static images
  - Video files
  - Live webcam feed
- Draws color-coded bounding boxes (amber = crack, red = pothole) with a live damage count overlay
- Adaptive thresholding makes detection robust to different lighting conditions and damage sizes, rather than relying on a single brightness cutoff

## How It Works

1. **Preprocessing** — convert to grayscale, apply CLAHE (adaptive contrast enhancement) to normalize uneven lighting, then a light Gaussian blur to reduce noise.
2. **Damage isolation** — adaptive thresholding flags any pixel that's notably darker than its local neighborhood, which catches both thin cracks and large potholes without needing a size-specific filter.
3. **Contour extraction** — morphological cleanup (open/close) removes small noise specks and connects nearby fragments into solid regions.
4. **Classification** — each detected region is classified by shape:
   - High aspect ratio + low circularity → **Crack**
   - Higher circularity → **Pothole**
5. **Annotation** — bounding boxes and labels are drawn on the frame, with a running count of each damage type.

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy

Install dependencies:
```bash
pip install opencv-python numpy
```

## Usage

**Image:**
```bash
python road_damage_detection.py --source image --input road.jpg --output annotated.jpg
```

**Video file:**
```bash
python road_damage_detection.py --source video --input drive.mp4 --output annotated.mp4
```

**Live webcam:**
```bash
python road_damage_detection.py --source webcam --camera 0
```
Press `q` to close the live preview window.

**Options:**
| Flag | Description |
|------|-------------|
| `--source` | `image`, `video`, or `webcam` (required) |
| `--input` | Path to input image/video file (required for image/video) |
| `--output` | Path to save annotated output |
| `--camera` | Webcam device index (default: `0`) |
| `--no-display` | Disable the live preview window (video/webcam mode) |

## Project Structure

```
.
├── road_damage_detection.py   # main script
└── README.md
```

## Limitations

- This is a classical CV approach (not deep learning), so it detects damage based on shape and contrast rather than learned visual features. It can be sensitive to:
  - Shadows, oil stains, or tar patches being misclassified as damage
  - Very low-contrast damage on already-dark road surfaces
- Detection thresholds (`ADAPTIVE_C`, `MIN_DAMAGE_AREA`, `ADAPTIVE_BLOCK_SIZE` in the script) may need tuning for different camera angles, road materials, or lighting conditions.

## Possible Future Improvements

- Train a deep learning model (e.g. YOLO) on a labeled road damage dataset for more robust detection
- Add severity scoring based on damage area relative to frame size
- Export detection results (counts, locations, timestamps) to a CSV/JSON log for tracking road conditions over time

## License

MIT License — feel free to use and modify.
