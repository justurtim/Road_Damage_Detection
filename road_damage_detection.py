"""
Road Damage Detection Vision System
=====================================
Detects potholes and cracks on road surfaces using classical computer vision
(no trained model / internet download required):

  1. Grayscale + CLAHE contrast enhancement (helps in uneven lighting)
  2. Black-hat morphological transform to isolate dark structures against
     the lighter road surface (both potholes and cracks are typically darker
     than surrounding asphalt/concrete)
  3. Adaptive thresholding + contour extraction on the black-hat result
  4. Shape-based classification of each contour:
       - Elongated, thin, low-circularity  -> Crack
       - Roughly round/blobby, higher area -> Pothole
       - Everything else that passes the area filter -> Damage (unclassified)
  5. Draws color-coded bounding boxes + a running count/severity summary

Works on:
  - a single image
  - a video file
  - a live webcam feed

Usage:
    python3 road_damage_detection.py --source image  --input path/to/image.jpg
    python3 road_damage_detection.py --source video  --input path/to/video.mp4
    python3 road_damage_detection.py --source webcam --camera 0

Output:
  - image mode: saves an annotated copy to --output (or annotated_output.jpg)
  - video/webcam mode: opens a display window (press 'q' to quit) and,
    if --output is given, saves an annotated video file
"""

import argparse
import sys
import numpy as np
import cv2


# --------------------------------------------------------------------------
# Tunable parameters
# --------------------------------------------------------------------------

MIN_DAMAGE_AREA = 150          # px^2 -- ignore tiny noise specks
ADAPTIVE_BLOCK_SIZE = 51       # must be odd; local neighborhood size for thresholding
ADAPTIVE_C = 10                 # how much darker than local mean to flag as damage
CRACK_ASPECT_RATIO_MIN = 2.5   # width/height (or vice versa) above this -> elongated
CRACK_CIRCULARITY_MAX = 0.35   # below this -> not round -> crack-like
POTHOLE_CIRCULARITY_MIN = 0.35 # at/above this -> round -> pothole-like

COLOR_CRACK = (0, 215, 255)     # amber (BGR)
COLOR_POTHOLE = (0, 0, 255)     # red (BGR)
COLOR_DAMAGE = (255, 150, 0)    # blue-ish (BGR), unclassified damage


# --------------------------------------------------------------------------
# Core pipeline
# --------------------------------------------------------------------------

def preprocess(frame):
    """Grayscale + CLAHE (adaptive contrast) + blur to reduce noise while
    keeping crack/pothole edges."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
    return blurred


def extract_damage_contours(gray):
    """Adaptive thresholding flags any pixel notably darker than its local
    neighborhood -- this catches both small thin cracks and large blobby
    potholes without needing a size-matched kernel, unlike a fixed-size
    black-hat transform."""
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_BLOCK_SIZE, C=ADAPTIVE_C
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                               np.ones((3, 3), np.uint8), iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                               np.ones((9, 9), np.uint8), iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def classify_contour(cnt):
    """Return ('Crack' | 'Pothole' | 'Damage', bbox, area) or None if the
    contour is too small to be meaningful."""
    area = cv2.contourArea(cnt)
    if area < MIN_DAMAGE_AREA:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    long_side, short_side = max(w, h), max(1, min(w, h))
    aspect_ratio = long_side / short_side

    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

    if aspect_ratio >= CRACK_ASPECT_RATIO_MIN and circularity <= CRACK_CIRCULARITY_MAX:
        label = "Crack"
    elif circularity >= POTHOLE_CIRCULARITY_MIN:
        label = "Pothole"
    else:
        label = "Damage"

    return label, (x, y, w, h), area


def process_frame(frame):
    """Run the full pipeline on a single BGR frame and return the
    annotated frame."""
    gray = preprocess(frame)
    contours = extract_damage_contours(gray)

    annotated = frame.copy()
    counts = {"Crack": 0, "Pothole": 0, "Damage": 0}

    for cnt in contours:
        result = classify_contour(cnt)
        if result is None:
            continue
        label, (x, y, w, h), area = result
        counts[label] += 1

        color = {"Crack": COLOR_CRACK, "Pothole": COLOR_POTHOLE,
                  "Damage": COLOR_DAMAGE}[label]

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, f"{label}", (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    summary = (f"Cracks: {counts['Crack']}  |  Potholes: {counts['Pothole']}  |  "
               f"Other damage: {counts['Damage']}")
    cv2.putText(annotated, summary, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated, counts


# --------------------------------------------------------------------------
# Entry points for each source type
# --------------------------------------------------------------------------

def run_on_image(input_path, output_path):
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"ERROR: could not read image '{input_path}'")
        sys.exit(1)

    annotated, counts = process_frame(frame)
    cv2.imwrite(output_path, annotated)
    print(f"Saved annotated image to: {output_path}")
    print(f"Detected -> Cracks: {counts['Crack']}, Potholes: {counts['Pothole']}, "
          f"Other damage: {counts['Damage']}")


def run_on_video(input_path, output_path, display=True):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video '{input_path}'")
        sys.exit(1)

    _write_stream(cap, output_path, display, window_name="Road Damage Detection - Video")


def run_on_webcam(camera_index, output_path, display=True):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: could not open webcam index {camera_index}")
        sys.exit(1)

    _write_stream(cap, output_path, display, window_name="Road Damage Detection - Webcam")


def _write_stream(cap, output_path, display, window_name):
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _ = process_frame(frame)

        if writer is not None:
            writer.write(annotated)

        if display:
            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video to: {output_path}")
    if display:
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Road Damage Detection Vision System")
    parser.add_argument("--source", choices=["image", "video", "webcam"], required=True,
                         help="Input source type")
    parser.add_argument("--input", help="Path to input image or video file")
    parser.add_argument("--output", help="Path to save annotated output (image/video)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--no-display", action="store_true",
                         help="Disable live preview window (video/webcam)")
    args = parser.parse_args()

    if args.source == "image":
        if not args.input:
            print("ERROR: --input is required for image mode")
            sys.exit(1)
        output_path = args.output or "annotated_output.jpg"
        run_on_image(args.input, output_path)

    elif args.source == "video":
        if not args.input:
            print("ERROR: --input is required for video mode")
            sys.exit(1)
        run_on_video(args.input, args.output, display=not args.no_display)

    elif args.source == "webcam":
        run_on_webcam(args.camera, args.output, display=not args.no_display)


if __name__ == "__main__":
    main()
