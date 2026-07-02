from pathlib import Path
import argparse
from functools import lru_cache

import joblib
import pandas as pd

from src.utils.feature_orchestrator import extract_all_features


@lru_cache(maxsize=4)
def load_model_bundle(model_path: str):
    return joblib.load(model_path)


def predict(image_path: str, model_path: str = "models/recapture_logistic.joblib") -> float:
    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    bundle = load_model_bundle(str(model_path))

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    features = extract_all_features(image_path)

    row = {}
    for col in feature_cols:
        row[col] = features.get(col, 0.0)

    X = pd.DataFrame([row], columns=feature_cols)

    score = float(model.predict_proba(X)[0][1])
    score = max(0.0, min(1.0, score))

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument(
        "--model",
        type=str,
        default="models/recapture_logistic.joblib",
    )

    args = parser.parse_args()

    score = predict(args.image_path, args.model)

    # Assignment wants only one number
    print(f"{score:.4f}")


if __name__ == "__main__":
    main()