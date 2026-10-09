import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib


FEATURES = [
    "time_sin",
    "time_cos",
    "day_of_week",
]


def add_features(df):
    df = df.copy()

    df["time_minutes"] = (
        df["Start"].dt.hour * 60 +
        df["Start"].dt.minute
    )

    df["time_sin"] = np.sin(
        2 * np.pi * df["time_minutes"] / 1440
    )

    df["time_cos"] = np.cos(
        2 * np.pi * df["time_minutes"] / 1440
    )

    df["day_of_week"] = df["Start"].dt.dayofweek

    return df


raw = pd.read_csv(
    "data.csv",
    usecols=["Type", "Start", "End", "Duration"]
)

raw["Start"] = pd.to_datetime(raw["Start"])
raw["End"] = pd.to_datetime(raw["End"])


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


model_df = df.dropna(
    subset=FEATURES + ["minutes_until_next_feed"]
).reset_index(drop=True)


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


joblib.dump(
    model,
    "feed_model.joblib"
)


print(f"Training rows: {len(model_df)}")
print("Model trained on all available data.")
print("Model saved to feed_model.joblib")