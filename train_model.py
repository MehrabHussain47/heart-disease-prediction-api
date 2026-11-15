import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_CANDIDATES = ["target", "num", "output", "heart_disease"]


def main(data_path: str, output_path: str) -> None:
    data = pd.read_csv(data_path)
    target = next((name for name in TARGET_CANDIDATES if name in data.columns), None)
    if target is None:
        raise ValueError(f"No target column found. Expected one of: {TARGET_CANDIDATES}")

    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected feature columns: {sorted(missing)}")

    x = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data[target], errors="coerce")
    valid = x.notna().all(axis=1) & y.notna()
    x, y = x.loc[valid], (y.loc[valid] > 0).astype(int)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    pipeline.fit(x, y)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": pipeline,
        "metadata": {
            "model_type": "LogisticRegression with StandardScaler",
            "features": FEATURES,
            "target_column": target,
        },
    }, output)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heart.csv")
    parser.add_argument("--output", default="model/heart_model.joblib")
    args = parser.parse_args()
    main(args.data, args.output)
