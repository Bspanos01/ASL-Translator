"""
Collect custom ASL landmark data from your webcam.

Usage:
    python3 data/collect.py

Controls:
    - Press a letter key (A-Z) to set the current label
    - Press SPACE to start/stop recording samples for that label
    - Press Q to quit and save

It captures normalized landmarks at ~30fps while recording,
then appends them to data/asl_dataset/custom_landmarks.csv
"""

import os
import sys
import csv
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hand_tracker import HandTracker
from ui.display import draw_landmarks, apply_mirror_flip


def main():
    tracker = HandTracker(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    # Build header
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")

    # Load existing custom data if any
    output_path = os.path.join(os.path.dirname(__file__), "asl_dataset", "custom_landmarks.csv")
    all_rows = []
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            all_rows = [row for row in reader]
        print(f"Loaded {len(all_rows)} existing custom samples")

    current_label = None
    recording = False
    session_count = 0
    label_counts = {}

    # Count existing samples per label
    for row in all_rows:
        lbl = row[-1]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print("=== ASL Data Collector ===")
    print("Press A-Z to select a letter")
    print("Press 1=del, 2=space, 3=nothing")
    print("Press SPACE to start/stop recording")
    print("Press Q to quit and save")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = apply_mirror_flip(frame)
        norm, raw = tracker.get_both(frame)

        # Draw hand skeleton
        draw_landmarks(frame, raw)

        # Status panel
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)

        if current_label:
            count = label_counts.get(current_label, 0)
            status = "RECORDING" if recording else "PAUSED"
            color = (0, 0, 255) if recording else (0, 180, 255)
            cv2.putText(frame, f"Label: {current_label}  |  {status}  |  Samples: {count}  |  Session: {session_count}",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "Press a letter key (A-Z) to select label",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Hand detected indicator
        if norm is not None:
            cv2.circle(frame, (w - 30, 30), 12, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (w - 30, 30), 12, (0, 0, 200), -1)

        # Record if active
        if recording and norm is not None and current_label is not None:
            row = [f"{v:.6f}" for v in norm] + [current_label]
            all_rows.append(row)
            session_count += 1
            label_counts[current_label] = label_counts.get(current_label, 0) + 1

        cv2.imshow("ASL Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break
        elif key == 32:  # SPACE
            if current_label:
                recording = not recording
                if recording:
                    print(f"  Recording '{current_label}'...")
                else:
                    print(f"  Paused. Got {session_count} samples this session.")
        elif key == ord('1'):
            current_label = "del"
            recording = False
            session_count = 0
            print(f"Selected: del")
        elif key == ord('2'):
            current_label = "space"
            recording = False
            session_count = 0
            print(f"Selected: space")
        elif key == ord('3'):
            current_label = "nothing"
            recording = False
            session_count = 0
            print(f"Selected: nothing")
        elif ord('a') <= key <= ord('z'):
            current_label = chr(key).upper()
            recording = False
            session_count = 0
            print(f"Selected: {current_label}")

    # Save
    cap.release()
    cv2.destroyAllWindows()

    if all_rows:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(all_rows)
        print(f"\nSaved {len(all_rows)} total samples to {output_path}")
        print("Label breakdown:")
        for lbl in sorted(label_counts.keys()):
            print(f"  {lbl}: {label_counts[lbl]}")
        print(f"\nNow retrain: python3 model/train.py")
    else:
        print("No samples collected.")


if __name__ == "__main__":
    main()
