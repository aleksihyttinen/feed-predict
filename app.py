import asyncio
import json
import os
import secrets
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "feed_model.joblib"
SAVE_SCRIPT = BASE_DIR / "save.py"
TEMP_DIR = Path("/tmp/my-model")

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
RETRAIN_SECRET = os.environ["RETRAIN_SECRET"]

app = FastAPI()

model = joblib.load(MODEL_PATH)
retrain_lock = asyncio.Lock()

FEATURES = [
    "time_sin",
    "time_cos",
    "day_of_week",
]


class FeedRequest(BaseModel):
    last_feed: datetime


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/predict")
def predict(request: FeedRequest):
    last_feed = pd.Timestamp(request.last_feed)

    time_minutes = last_feed.hour * 60 + last_feed.minute

    row = pd.DataFrame([{
        "time_sin": np.sin(
            2 * np.pi * time_minutes / 1440
        ),
        "time_cos": np.cos(
            2 * np.pi * time_minutes / 1440
        ),
        "day_of_week": last_feed.dayofweek,
    }])

    current_model = model

    predicted_minutes = float(
        current_model.predict(row[FEATURES])[0]
    )

    next_feed = (
        last_feed + pd.Timedelta(minutes=predicted_minutes)
    ).floor("min")

    return {
        "predicted_minutes": round(predicted_minutes, 1),
        "predicted_next_feed": next_feed.isoformat(),
    }


@app.post("/retrain")
async def retrain(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    global model

    if (
        authorization is None
        or not authorization.startswith("Bearer ")
        or not secrets.compare_digest(
            authorization.removeprefix("Bearer "),
            RETRAIN_SECRET,
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Virheellinen salasana.",
        )

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Upload a CSV file.",
        )

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(
            prefix="training-",
            dir=TEMP_DIR,
        ) as temp_dir:
            csv_path = Path(temp_dir) / "training_data.csv"

            size = 0

            with csv_path.open("wb") as destination:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break

                    size += len(chunk)

                    if size > MAX_UPLOAD_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="CSV exceeds the 50 MB limit.",
                        )

                    destination.write(chunk)

            if size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="CSV file is empty.",
                )

            async with retrain_lock:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(SAVE_SCRIPT),
                    str(csv_path),
                    cwd=str(BASE_DIR),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=600,
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    raise HTTPException(
                        status_code=504,
                        detail="Model retraining timed out.",
                    )

                if process.returncode != 0:
                    print(
                        "Model retraining failed:",
                        stderr.decode(errors="replace"),
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="Model retraining failed.",
                    )

                try:
                    summary = json.loads(stdout.decode("utf-8"))
                except json.JSONDecodeError:
                    print(
                        "Invalid JSON from training script:",
                        repr(stdout.decode("utf-8")),
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="Training returned an invalid response.",
                    )

                try:
                    new_model = joblib.load(MODEL_PATH)
                except Exception:
                    print("Could not load the newly trained model.")
                    raise HTTPException(
                        status_code=500,
                        detail="Could not load the newly trained model.",
                    )

                model = new_model
                return summary

    finally:
        await file.close()