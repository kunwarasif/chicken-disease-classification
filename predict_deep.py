"""
Predict disease classes for images using the trained MLP model.

This script loads the saved model (`mlp_model.pkl`) produced by
`train_deep_model.py` and applies it to one or more input images. Images are
converted to grayscale, resized to 8×8 pixels, flattened and scaled before
being passed to the classifier. Predictions are printed to the console.

Usage:

```
python predict_deep.py <image1> [<image2> ...]
```

Example:

```
python predict_deep.py sample1.png sample2.jpg
```

Ensure you have run `train_deep_model.py` first so that `mlp_model.pkl` exists.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import joblib


def preprocess_image(path: Path, size=(8, 8)) -> np.ndarray:
    """Load and preprocess an image for prediction."""
    img = Image.open(path).convert("L")
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_deep.py <image1> [<image2> ...]")
        sys.exit(1)
    # Load model and label encoder
    try:
        clf, le = joblib.load("mlp_model.pkl")
    except FileNotFoundError:
        print("Model file mlp_model.pkl not found. Run train_deep_model.py first.")
        sys.exit(1)
    image_paths = [Path(p) for p in sys.argv[1:]]
    features = []
    valid_paths = []
    for path in image_paths:
        if path.exists():
            try:
                features.append(preprocess_image(path))
                valid_paths.append(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")
        else:
            print(f"File not found: {path}")
    if not features:
        print("No valid images to process.")
        sys.exit(1)
    X = np.vstack(features)
    preds = clf.predict(X)
    pred_labels = le.inverse_transform(preds)
    for path, label in zip(valid_paths, pred_labels):
        print(f"{path}: {label}")


if __name__ == "__main__":
    main()