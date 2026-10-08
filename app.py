from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel


app = FastAPI()

model = joblib.load("feed_model.joblib")

FEATURES = [
    "time_sin",
    "time_cos",
    "day_of_week",
]


class FeedRequest(BaseModel):
    last_feed: datetime


@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/predict")
def predict(request: FeedRequest):
    last_feed = pd.Timestamp(request.last_feed)

    time_minutes = (
        last_feed.hour * 60 +
        last_feed.minute
    )

    row = pd.DataFrame([{
        "time_sin": np.sin(
            2 * np.pi * time_minutes / 1440
        ),
        "time_cos": np.cos(
            2 * np.pi * time_minutes / 1440
        ),
        "day_of_week": last_feed.dayofweek,
    }])

    predicted_minutes = float(
        model.predict(row[FEATURES])[0]
    )

    next_feed = (
        last_feed +
        pd.Timedelta(minutes=predicted_minutes)
    ).floor("min")

    return {
        "predicted_minutes": round(predicted_minutes, 1),
        "predicted_next_feed": next_feed.isoformat(),
    }
