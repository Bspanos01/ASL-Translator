"""
Collect custom ASL landmark data from your webcam.

Usage:
    python3 data/collect.py

You pick a class from the menu BEFORE the webcam starts.
Then SPACE to record, SPACE to pause, N for next class, Q to quit.
"""

import os
import sys
import csv

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hand_tracker import HandTracker
from ui.display import draw_landmarks, apply_mirror_flip

# All available classes (26 letters + ILY)
LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
ALL_CLASSES = LETTERS + ["ILY", "CONFIRM"]


def pick_class():
    """Terminal menu to pick a class. Returns class name or None to quit."""
    print("\n" + "=" * 55)
    print("          SELECT A CLASS TO COLLECT")
    print("=" * 55)

    print("\n  LETTERS:")
    for i, l in enumerate(LETTERS):
        print(f"    {i + 1:2d}) {l}", end="")
        if (i + 1) % 9 == 0:
            print()
    print()

    print(f"\n    27) ILY      (I Love You)")
    print(f"    28) CONFIRM  (Thumbs Up — confirms word)")

    print("\n  Type a number, class name, or 'q' to quit.")
    print("=" * 55)

    choice = input("\n  > ").strip()

    if choice.lower() == 'q':
        return None

    # Try as menu number
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(ALL_CLASSES):
            return ALL_CLASSES[idx]
    except ValueError:
        pass

    # Try as class name
    for cls in ALL_CLASSES:
        if cls.lower() == choice.lower():
            return cls

    print(f"  Unknown: '{choice}'")
    return pick_class()


def main():
    # Pick class first, before webcam
    current_label = pick_class()
    if current_label is None:
        print("Exiting.")
        return

    print(f"\n  Collecting for: {current_label}")
    print("  Webcam opening... Controls:")
    print("    SPACE = start/stop recording")
    print("    N     = pick a new class (pauses, returns to menu)")
    print("    Q     = quit and save\n")

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
        print(f"  Loaded {len(all_rows)} existing custom samples")

    recording = False
    session_count = 0
    label_counts = {}

    # Count existing samples per label
    for row in all_rows:
        lbl = row[-1]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = apply_mirror_flip(frame)
        norm, raw = tracker.get_both(frame)

        draw_landmarks(frame, raw)

        # Status panel
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)

        count = label_counts.get(current_label, 0)
        status = "RECORDING" if recording else "READY"
        color = (0, 0, 255) if recording else (0, 180, 255)
        cv2.putText(frame,
                    f"Class: {current_label}  |  {status}  |  Total: {count}  |  Session: {session_count}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Bottom hint
        cv2.rectangle(frame, (0, h - 35), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, "SPACE=record/pause   N=new class   Q=save & quit",
                    (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Hand detected indicator
        if norm is not None:
            cv2.circle(frame, (w - 30, 30), 12, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (w - 30, 30), 12, (0, 0, 200), -1)

        # Record if active
        if recording and norm is not None:
            row = [f"{v:.6f}" for v in norm] + [current_label]
            all_rows.append(row)
            session_count += 1
            label_counts[current_label] = label_counts.get(current_label, 0) + 1

        cv2.imshow("ASL Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break

        elif key == 32:  # SPACE
            recording = not recording
            if recording:
                print(f"  Recording '{current_label}'...")
            else:
                print(f"  Paused. {session_count} samples this session.")

        elif key in (ord('n'), ord('N')):
            recording = False
            print(f"  Paused. {session_count} samples this session.")
            new_label = pick_class()
            if new_label is None:
                break
            current_label = new_label
            session_count = 0
            print(f"  Now collecting: {current_label}")

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
        print(f"\nRetrain: python3 model/train.py")
    else:
        print("No samples collected.")


if __name__ == "__main__":
    main()
