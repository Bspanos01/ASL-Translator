import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_landmarks(self, frame):
        """Takes a BGR OpenCV frame, returns flat list of 63 floats or None."""
        rgb = frame[:, :, ::-1]  # BGR to RGB
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return landmarks
        return None

    def normalize_landmarks(self, landmarks):
        """Position- and scale-invariant normalization of 63-float landmark list."""
        if landmarks is None:
            return None
        pts = np.array(landmarks).reshape(21, 3)
        wrist = pts[0].copy()
        pts = pts - wrist  # position invariant
        scale = np.linalg.norm(pts[9]) + 1e-6  # distance wrist(0) to middle MCP(9)
        pts = pts / scale  # scale invariant
        return pts.flatten().tolist()

    def get_both(self, frame):
        """Returns (normalized_63_floats, raw_mediapipe_landmarks) in one call."""
        rgb = frame[:, :, ::-1]
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            normalized = self.normalize_landmarks(landmarks)
            return normalized, hand
        return None, None
