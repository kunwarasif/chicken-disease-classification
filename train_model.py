"""
Train a RandomForest classifier for chicken disease classification.

This script is designed to be flexible: if a `data.csv` file is present in the
working directory, it will be used as the training dataset. The CSV should
contain only numeric feature columns and a final column named `target` (the
class label). If `data.csv` is absent, the script falls back to the
`wine` dataset from scikit‑learn for demonstration purposes. The trained
model is saved to `model.joblib`, a classification report is written to
`classification_report.txt`, and visualizations of the confusion matrix and
feature importance are saved as PNG files.

To run this script:

```
python train_model.py
```

Dependencies are listed in `requirements.txt`.
"""

import os
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib
matplotlib.use('Agg')  # Use a non‑interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def load_dataset():
    """
    Load dataset from `data.csv` if present; otherwise use scikit‑learn's wine dataset.

    Returns
    -------
    X : pandas.DataFrame or ndarray
        Feature matrix.
    y : pandas.Series or ndarray
        Target vector.
    feature_names : list of str
        List of feature names.
    class_names : list of str
        List of class names.
    """
    if os.path.exists("data.csv"):
        df = pd.read_csv("data.csv")
        # Assume the last column is the target label
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        feature_names = list(X.columns)
        # Derive class names from unique labels
        unique = sorted(unique_labels(y))
        class_names = [str(cls) for cls in unique]
        return X, y, feature_names, class_names
    else:
        data = load_wine()
        return data.data, data.target, list(data.feature_names), list(data.target_names)


def train_and_evaluate(X, y, feature_names, class_names):
    """
    Train a RandomForest classifier and evaluate it on a holdout test set.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    feature_names : list of str
        List of feature names.
    class_names : list of str
        List of class names.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)
    # Write classification report to file
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)
    # Save model for future predictions
    joblib.dump(clf, "model.joblib")
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    plt.close()
    # Plot feature importances
    importances = clf.feature_importances_
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.title("Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", bbox_inches="tight")
    plt.close()


def main():
    X, y, feature_names, class_names = load_dataset()
    train_and_evaluate(X, y, feature_names, class_names)


if __name__ == "__main__":
    main()