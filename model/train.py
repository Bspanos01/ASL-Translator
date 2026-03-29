import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

LABEL_MAP = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
    'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16,
    'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
    'Z': 25, 'ILY': 26, 'CONFIRM': 27, 'del': 28, 'nothing': 29, 'space': 30,
}


def load_dataset(data_dir):
    """Scan data_dir for CSV files and load features + labels."""
    all_X = []
    all_y = []

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Found {len(csv_files)} CSV file(s) in {data_dir}")

    for csv_file in tqdm(csv_files, desc="Loading CSVs"):
        path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(path)

        # Detect label column
        label_col = None
        for candidate in ['label', 'class', 'Letter']:
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            label_col = df.columns[-1]

        y = df[label_col].values
        X = df.drop(columns=[label_col]).values.astype(np.float32)

        all_X.append(X)
        all_y.append(y)

    X_array = np.vstack(all_X)
    y_array = np.concatenate(all_y)

    print(f"Total samples: {len(y_array)}")
    unique, counts = np.unique(y_array, return_counts=True)
    print("Class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt}")

    return X_array, y_array


def preprocess(X, y):
    """Encode labels and split into train/test sets."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, le


def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    print("Training RandomForest...")
    start = time.time()
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=4,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Training complete in {elapsed:.1f}s")
    return clf


def save_model(model, encoder, out_path):
    """Save model and encoder as a pickle dict."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "encoder": encoder}, f)
    print(f"Model saved to {out_path}")


def main():
    data_dir = "data/asl_dataset"
    out_path = "model/asl_model.pkl"

    X, y = load_dataset(data_dir)
    X_train, X_test, y_train, y_test, le = preprocess(X, y)
    clf = train_model(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    all_labels = list(range(len(le.classes_)))
    print(classification_report(y_test, y_pred, labels=all_labels, target_names=le.classes_, zero_division=0))

    save_model(clf, le, out_path)


if __name__ == "__main__":
    main()
