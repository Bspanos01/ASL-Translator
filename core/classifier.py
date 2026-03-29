import pickle
import numpy as np


class ASLClassifier:
    def __init__(self, model_path="model/asl_model.pkl"):
        self.labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["ILY", "CONFIRM", "del", "nothing", "space"]
        self.model = None
        self.encoder = None
        self.load_model(model_path)

    def load_model(self, path):
        """Load model and encoder from pickle file."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.encoder = data["encoder"]
            print(f"[Classifier] Model loaded from {path}")
        except FileNotFoundError:
            print(f"[Classifier] Model file not found: {path}")
            self.model = None
            self.encoder = None
        except Exception as e:
            print(f"[Classifier] Failed to load model: {e}")
            self.model = None
            self.encoder = None

    def predict_letter(self, landmarks):
        """Predict a single letter from 63-float normalized landmarks."""
        if self.model is None or self.encoder is None or landmarks is None:
            return None, 0.0
        try:
            X = np.array(landmarks).reshape(1, 63)
            proba = self.model.predict_proba(X)[0]
            idx = np.argmax(proba)
            confidence = float(proba[idx])
            if confidence < 0.35:
                return None, 0.0
            letter = self.encoder.inverse_transform([idx])[0]
            if str(letter) in ("nothing", "space", "del"):
                return None, 0.0
            return str(letter), confidence
        except Exception as e:
            print(f"[Classifier] Prediction error: {e}")
            return None, 0.0

    def get_top_predictions(self, landmarks, n=3):
        """Return top N (letter, confidence) tuples sorted descending."""
        if self.model is None or self.encoder is None or landmarks is None:
            return []
        try:
            X = np.array(landmarks).reshape(1, 63)
            proba = self.model.predict_proba(X)[0]
            top_indices = np.argsort(proba)[::-1][:n]
            results = []
            for idx in top_indices:
                letter = self.encoder.inverse_transform([idx])[0]
                results.append((str(letter), float(proba[idx])))
            return results
        except Exception as e:
            print(f"[Classifier] Top predictions error: {e}")
            return []

    def is_ready(self):
        """Returns True if model and encoder are both loaded."""
        return self.model is not None and self.encoder is not None
