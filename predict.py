"""
Predict chicken disease classes using a trained model.

This script loads a RandomForest model saved by `train_model.py` (model.joblib)
and applies it to a CSV file of new samples. The input CSV should contain
feature columns matching those used during training (i.e., the same order and
number of columns as `data.csv` without the `target` column). The script
outputs predictions to standard output and saves them to `predictions.csv`.

Usage:

```
python predict.py <input_csv>
```

Example:

```
python predict.py new_samples.csv
```
"""

import sys
import pandas as pd
import joblib


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_csv>")
        sys.exit(1)
    input_file = sys.argv[1]
    try:
        model = joblib.load("model.joblib")
    except FileNotFoundError:
        print("Error: model.joblib not found. Train the model first using train_model.py.")
        sys.exit(1)
    data = pd.read_csv(input_file)
    # Ensure we only pass feature columns (no target)
    preds = model.predict(data)
    # Create a DataFrame for predictions and save
    pred_df = pd.DataFrame({"prediction": preds})
    output_path = "predictions.csv"
    pred_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(pred_df)


if __name__ == "__main__":
    main()