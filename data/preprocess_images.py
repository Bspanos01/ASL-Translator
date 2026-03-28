"""
Preprocess ASL Alphabet image dataset into landmark CSVs.

Usage:
    1. Download the ASL Alphabet dataset from Kaggle:
       https://www.kaggle.com/datasets/grassknoted/asl-alphabet
    2. Extract it so the folder structure looks like:
       data/asl_alphabet_train/
           A/  (images)
           B/  (images)
           ...
           Z/
           del/
           nothing/
           space/
    3. Run this script:
       python3 data/preprocess_images.py

It will process every image through MediaPipe, extract 63 landmark features
(21 landmarks x 3 coords), and save them as data/asl_dataset/landmarks.csv
"""

import os
import sys
import csv

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hand_tracker import HandTracker


def preprocess_dataset(image_dir, output_csv):
    """Process all images in image_dir subfolders and write landmarks to CSV."""
    tracker = HandTracker(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Build header: x0, y0, z0, x1, y1, z1, ..., x20, y20, z20, label
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")

    # Find all class subdirectories
    classes = sorted([
        d for d in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, d))
    ])

    if not classes:
        print(f"ERROR: No subdirectories found in {image_dir}")
        print("Expected folders like A/, B/, ..., Z/, del/, nothing/, space/")
        return

    print(f"Found {len(classes)} classes: {classes}")

    rows = []
    skipped = 0
    total = 0

    for label in classes:
        class_dir = os.path.join(image_dir, label)
        images = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        print(f"Processing {label}: {len(images)} images")

        for img_file in tqdm(images, desc=f"  {label}", leave=False):
            total += 1
            img_path = os.path.join(class_dir, img_file)
            frame = cv2.imread(img_path)
            if frame is None:
                skipped += 1
                continue

            landmarks = tracker.extract_landmarks(frame)
            if landmarks is None:
                skipped += 1
                continue

            normalized = tracker.normalize_landmarks(landmarks)
            if normalized is None or len(normalized) != 63:
                skipped += 1
                continue

            rows.append(normalized + [label])

    # Write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nDone! Processed {total} images.")
    print(f"  Successful landmarks: {len(rows)}")
    print(f"  Skipped (no hand detected): {skipped}")
    print(f"  Saved to: {output_csv}")


if __name__ == "__main__":
    # Default paths — adjust if your dataset is elsewhere
    image_dir = os.path.join(os.path.dirname(__file__), "asl_alphabet_train", "asl_alphabet_train")
    output_csv = os.path.join(os.path.dirname(__file__), "asl_dataset", "landmarks.csv")

    # Also check one level up in case of different extraction structure
    if not os.path.isdir(image_dir):
        image_dir = os.path.join(os.path.dirname(__file__), "asl_alphabet_train")

    if not os.path.isdir(image_dir):
        print("Dataset not found! Please download from:")
        print("  https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print()
        print("Then extract it into the data/ folder so it looks like:")
        print("  data/asl_alphabet_train/A/  (or data/asl_alphabet_train/asl_alphabet_train/A/)")
        print("  data/asl_alphabet_train/B/")
        print("  ...")
        sys.exit(1)

    preprocess_dataset(image_dir, output_csv)
