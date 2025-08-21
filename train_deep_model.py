"""
Train a deep learning classifier (MLP) for image-based chicken disease classification.

This script attempts to load images from an `images/` directory where each
subdirectory corresponds to a disease class and contains image files. If such
a directory is not found, it falls back to the built-in digits dataset from
scikit-learn (using only digits 0, 1 and 2) as a placeholder. Images are
resized to 8×8 pixels and converted to grayscale. Features are flattened and
scaled before training an MLP classifier with two hidden layers.

Outputs:
 - `mlp_model.pkl`: saved scikit-learn model via joblib.
 - `mlp_classification_report.txt`: classification report with accuracy and metrics.
 - `mlp_confusion_matrix.png`: confusion matrix visualization.

Usage:

```
python train_deep_model.py
```

You can customize the `images/` directory with your own images. Each class
should be a subfolder inside `images/` with PNG/JPG images.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.datasets import load_digits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def load_custom_images(image_dir: str, target_size=(8, 8)):
    """
    Load images from a directory structure for classification.

    Parameters
    ----------
    image_dir : str
        Path to the top-level directory containing subdirectories for each class.
    target_size : tuple of int
        Desired size (width, height) for resizing images.

    Returns
    -------
    X : np.ndarray
        Flattened image data.
    y : list
        Corresponding class labels.
    class_names : list of str
        Sorted list of unique class labels.
    """
    images = []
    labels = []
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists() or not image_dir_path.is_dir():
        return None, None, None
    for class_dir in sorted(d.name for d in image_dir_path.iterdir() if d.is_dir()):
        class_path = image_dir_path / class_dir
        for file in class_path.iterdir():
            if file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
                # Convert image to grayscale and resize
                try:
                    img = Image.open(file).convert("L")
                    img = img.resize(target_size)
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                    images.append(arr.flatten())
                    labels.append(class_dir)
                except Exception:
                    # Skip unreadable images
                    continue
    if images:
        X = np.array(images)
        y = np.array(labels)
        class_names = sorted(unique_labels(y))
        return X, y, class_names
    else:
        return None, None, None


def load_digits_subset():
    """
    Load a subset of the scikit-learn digits dataset (classes 0,1,2) as a fallback.

    Returns
    -------
    X : np.ndarray
        Flattened image data.
    y : np.ndarray
        Corresponding class labels (as strings for consistency with custom images).
    class_names : list of str
        Sorted list of unique class names.
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    mask = y < 3  # select classes 0,1,2
    X = X[mask] / 16.0  # scale pixel values to [0,1]
    y = y[mask].astype(str)  # convert to strings for label encoder
    class_names = sorted(unique_labels(y))
    return X, y, class_names


def train_model(X: np.ndarray, y: np.ndarray, class_names: list, hidden_layers=(100, 50)):
    """
    Train an MLP classifier on the provided data.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels (string names).
    class_names : list of str
        Unique class labels.
    hidden_layers : tuple of int
        Sizes of hidden layers.
    """
    # Encode class names to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # Save model and label encoder together in a tuple
    joblib.dump((clf, le), "mlp_model.pkl")
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    with open("mlp_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("MLP Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("mlp_confusion_matrix.png", bbox_inches="tight")
    plt.close()
    return acc


def main():
    # Try loading custom images
    X, y, class_names = load_custom_images("images")
    if X is None or len(class_names) == 0:
        # Fall back to digits dataset
        X, y, class_names = load_digits_subset()
        print("No custom images found. Using digits dataset (classes 0,1,2) as placeholder.")
    print(f"Loaded {len(X)} samples across {len(class_names)} classes: {class_names}")
    accuracy = train_model(X, y, class_names)
    print(f"Test accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()