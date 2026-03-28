import cv2
import numpy as np
import mediapipe as mp

# Colors (BGR)
GREEN = (0, 255, 180)
WHITE = (255, 255, 255)
DARK_BG = (20, 20, 20)
AMBER = (0, 180, 255)
RED = (0, 80, 220)
GRAY = (140, 140, 140)
LIGHT_PURPLE = (220, 180, 255)

_mp_drawing = mp.solutions.drawing_utils
_mp_hands = mp.solutions.hands

_landmark_spec = _mp_drawing.DrawingSpec(color=GREEN, thickness=2, circle_radius=3)
_connection_spec = _mp_drawing.DrawingSpec(color=WHITE, thickness=1)


def draw_landmarks(frame, raw_landmarks):
    """Draw hand skeleton overlay using MediaPipe drawing utils."""
    if raw_landmarks is not None:
        _mp_drawing.draw_landmarks(
            frame,
            raw_landmarks,
            _mp_hands.HAND_CONNECTIONS,
            _landmark_spec,
            _connection_spec,
        )


def draw_letter_overlay(frame, letter, confidence, top_predictions=None):
    """Draw predicted letter with confidence bar in bottom-left, above the word panel."""
    h, w = frame.shape[:2]
    box_top = h - 260
    box_bottom = h - 135

    # Semi-transparent dark box
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, box_top), (200, box_bottom), DARK_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if letter:
        # Large letter
        cv2.putText(frame, str(letter), (30, box_top + 65), cv2.FONT_HERSHEY_SIMPLEX, 3.0, GREEN, 4)
        # Confidence bar background
        cv2.rectangle(frame, (30, box_top + 80), (180, box_top + 95), (60, 60, 60), -1)
        # Confidence bar fill
        bar_width = int(150 * confidence)
        cv2.rectangle(frame, (30, box_top + 80), (30 + bar_width, box_top + 95), GREEN, -1)
        # Confidence text
        cv2.putText(frame, f"{int(confidence * 100)}%", (135, box_top + 93), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)

    # Show 2nd and 3rd predictions
    if top_predictions and len(top_predictions) > 1:
        for i, (pred_letter, pred_conf) in enumerate(top_predictions[1:3], start=1):
            y_pos = box_top + 100 + (i - 1) * 15
            cv2.putText(frame, f"{pred_letter}: {int(pred_conf * 100)}%", (30, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)


def draw_word_panel(frame, buffer_str, suggestions, word_count=0):
    """Draw word panel across the bottom 130px of the frame."""
    h, w = frame.shape[:2]
    panel_top = h - 130

    # Dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_top), (w, h), DARK_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Line 1: current word being signed
    display_word = buffer_str + "_" if buffer_str else "_"
    cv2.putText(frame, f"Signing: {display_word}", (20, panel_top + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    # Word count
    if word_count > 0:
        cv2.putText(frame, f"Words: {word_count}", (w - 140, panel_top + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)

    # Line 2: suggestions
    cv2.putText(frame, "Suggest:", (20, panel_top + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
    x_offset = 120
    for i, suggestion in enumerate(suggestions[:3]):
        color = AMBER if i == 0 else GRAY
        text = f"[{i + 1}] {suggestion}"
        cv2.putText(frame, text, (x_offset, panel_top + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        x_offset += len(text) * 14 + 20

    # Line 3: controls hint
    cv2.putText(frame, "SPACE=confirm  D=delete  1/2/3=pick  S=speak  J/Z=letter  C=clear  X=undo word",
                (20, panel_top + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1)


def draw_sentence(frame, sentence):
    """Draw confirmed sentence at the top of the frame."""
    if not sentence:
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), DARK_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, sentence, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, LIGHT_PURPLE, 2)


def draw_status_bar(frame, wili_connected, model_loaded):
    """Draw status indicators in the bottom-right corner."""
    h, w = frame.shape[:2]
    # WiLi status
    wili_text = "WiLi: ON" if wili_connected else "WiLi: OFF"
    wili_color = GREEN if wili_connected else RED
    cv2.putText(frame, wili_text, (w - 170, h - 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, wili_color, 1)
    # Model status
    model_text = "Model: OK" if model_loaded else "Model: --"
    model_color = GREEN if model_loaded else RED
    cv2.putText(frame, model_text, (w - 170, h - 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, model_color, 1)


def draw_candidate_progress(frame, candidate_letter, progress_float):
    """Draw progress indicator for current candidate letter in top-right."""
    if candidate_letter is None:
        return
    h, w = frame.shape[:2]
    center_x = w - 50
    center_y = 50
    radius = 25

    # Background circle
    cv2.circle(frame, (center_x, center_y), radius, (60, 60, 60), 2)
    # Progress arc
    angle = int(360 * progress_float)
    if angle > 0:
        cv2.ellipse(frame, (center_x, center_y), (radius, radius), -90, 0, angle, GREEN, 3)
    # Candidate letter
    cv2.putText(frame, str(candidate_letter), (center_x - 8, center_y + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRAY, 2)


def apply_mirror_flip(frame):
    """Flip frame horizontally for natural webcam feel."""
    return cv2.flip(frame, 1)
