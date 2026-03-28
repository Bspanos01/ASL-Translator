import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from model.train import load_dataset, preprocess


def load_model_and_data(model_path, data_dir):
    """Load the saved model and a test dataset."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    encoder = data["encoder"]

    X, y = load_dataset(data_dir)
    _, X_test, _, y_test, _ = preprocess(X, y)
    return model, encoder, X_test, y_test


def plot_confusion_matrix(cm, classes):
    """Plot and save a confusion matrix as a PNG."""
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title("ASL Classifier — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path = "model/confusion_matrix.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {out_path}")


def print_classification_report(model, encoder, X_test, y_test):
    """Print per-class precision, recall, F1."""
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    return y_pred


def main():
    model_path = "model/asl_model.pkl"
    data_dir = "data/asl_dataset"

    model, encoder, X_test, y_test = load_model_and_data(model_path, data_dir)
    y_pred = print_classification_report(model, encoder, X_test, y_test)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, encoder.classes_)


if __name__ == "__main__":
    main()
