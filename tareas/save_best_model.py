"""Train three classifiers, pick the best by accuracy and export the pipeline.

This script mirrors the models used in the notebook `training_model.ipynb`:
- LogisticRegression
- SVC (rbf, C=1.0, gamma=3.0)
- DecisionTreeClassifier

It creates a preprocessing pipeline for numeric columns (median imputer + StandardScaler),
trains models, evaluates on a held-out test split, selects the best by accuracy and
saves the fitted pipeline (preprocessor + model) under `./models/best_model/`.

Usage:
    python tareas/save_best_model.py --csv /path/to/data_credit_risk.csv

If no --csv is provided, it tries 'data_credit_risk.csv' in the current dir.
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def build_preprocessor(X_train):
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    # Drop categorical columns as in the notebook
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", "drop", cat_cols),
        ],
        remainder="drop",
    )
    return preprocess, num_cols


def main(csv_path: Path, out_dir: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Reading {csv_path}")
    df = pd.read_csv(csv_path)

    if "default" not in df.columns:
        raise KeyError("The dataset must contain a 'default' column as target.")

    # Basic cleanup as in the notebook
    if "age" in df.columns:
        df["age"] = df["age"].abs()

    X = df.drop(columns=["default"])
    y = df["default"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess, num_cols = build_preprocessor(X_train)

    # Candidate models
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "SVM_RBF": SVC(kernel="rbf", C=1.0, gamma=3.0),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
    }

    results = {}

    for name, model in candidates.items():
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = {"pipeline": pipe, "accuracy": acc}
        print(f"{name} accuracy: {acc:.4f}")

    # Select best
    best_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
    best_acc = results[best_name]["accuracy"]
    best_pipe = results[best_name]["pipeline"]

    print(f"Best model: {best_name} (accuracy={best_acc:.4f})")

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model_pipeline.joblib"
    joblib.dump(best_pipe, model_path)
    print(f"Saved best pipeline to: {model_path}")

    # Save metadata
    meta = {"model_name": best_name, "accuracy": float(best_acc), "num_features": num_cols}
    joblib.dump(meta, out_dir / "metadata.joblib")
    print(f"Saved metadata to: {out_dir / 'metadata.joblib'}")

    # Example of loading and using the pipeline
    print("Example predict: ")
    sample = X_test.iloc[:3].astype(str).fillna("")
    loaded = joblib.load(model_path)
    print(loaded.predict(sample))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data_credit_risk.csv", help="Path to CSV dataset")
    parser.add_argument("--out", type=str, default="models/best_model", help="Output directory for exported model")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    main(csv_path, out_dir)
