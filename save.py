import json
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "feed_model.joblib"

FEATURES = [
    "time_sin",
    "time_cos",
    "day_of_week",
]


def add_features(df):
    df = df.copy()

    df["time_minutes"] = (
        df["Start"].dt.hour * 60
        + df["Start"].dt.minute
    )

    df["time_sin"] = np.sin(
        2 * np.pi * df["time_minutes"] / 1440
    )

    df["time_cos"] = np.cos(
        2 * np.pi * df["time_minutes"] / 1440
    )

    df["day_of_week"] = df["Start"].dt.dayofweek

    return df


def train_model(csv_path):
    raw = pd.read_csv(
        csv_path,
        usecols=["Type", "Start", "End"],
    )

    raw["Start"] = pd.to_datetime(raw["Start"], errors="coerce")
    raw["End"] = pd.to_datetime(raw["End"], errors="coerce")

    df = raw[
        raw["Type"] == "Feed"
    ].dropna(subset=["Start"]).copy()

    df = (
        df.sort_values(["Start", "End"], na_position="last")
        .drop_duplicates(subset="Start", keep="first")
        .reset_index(drop=True)
    )

    df["minutes_until_next_feed"] = (
        df["Start"].shift(-1) - df["Start"]
    ).dt.total_seconds() / 60

    df = add_features(df)

    model_df = (
        df.dropna(
            subset=FEATURES + ["minutes_until_next_feed"]
        )
        .reset_index(drop=True)
    )

    if len(model_df) < 2:
        raise ValueError(
            "Not enough valid feeding records to train the model."
        )

    if (model_df["minutes_until_next_feed"] <= 0).any():
        raise ValueError(
            "Training data contains non-positive feeding intervals."
        )

    X = model_df[FEATURES]
    y = model_df["minutes_until_next_feed"]

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=3,
        min_samples_leaf=10,
        loss="absolute_error",
        random_state=42,
    )

    model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=".feed_model-",
        suffix=".joblib",
        dir=MODEL_DIR,
    )
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        joblib.dump(model, temp_path)

        os.replace(temp_path, MODEL_PATH)
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "status": "success",
        "training_rows": len(model_df),
        "model_type": "GradientBoostingRegressor",
        "features": FEATURES,
        "model_file": "models/feed_model.joblib",
        "message": "Model trained on all available data.",
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python save.py <csv_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        summary = train_model(sys.argv[1])
        print(json.dumps(summary))
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)