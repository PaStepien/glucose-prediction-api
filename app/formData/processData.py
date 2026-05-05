"""
──────────────────────────────────────────────────────────────────────
Inference pipeline that exactly mirrors the training code in model.py.

Critical details extracted from training:
  - StandardScaler (NOT MinMax), two separate scalers: scaler_X, scaler_y
  - scaler_X fitted on features only (excludes patient_id, timestamp, glucose)
  - scaler_y fitted on glucose column only
  - TIME_STEPS = 36  →  3 hours of history (36 × 5 min)
  - horizon   = 6   →  predicts glucose 30 min ahead (6 × 5 min)
  - Input to LSTM: (batch, 36, 12)  — 11 features + glucose as col 12
  - Output: scaled glucose scalar → inverse_transform → mg/dL

Save your scalers after training with:
    import joblib
    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")
"""

import numpy as np
import pandas as pd
import joblib
import keras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import List
from pathlib import Path

app = FastAPI()

# ── Constants (must match training exactly) ───────────────────────────

TIME_STEPS   = 36          # 3 hours of history
HORIZON      = 6           # predict 30 min ahead
STEP_FREQ    = "5min"
INTERP_LIMIT = 2           # max 2 consecutive missing CGM steps filled

# Column order must match training — features list is everything except
# patient_id, timestamp, glucose. Glucose is appended last.
FEATURE_COLS = [
    "bolus_raw",
    "insulin_activity",
    "carbs",
    "meal_Breakfast",
    "meal_Dinner",
    "meal_HypoCorrection",
    "meal_Lunch",
    "meal_Snack",
    "meal_activity",
    "steps",
    "steps_weighted_avg",
]
# LSTM sees these 11 features + glucose as the 12th column → shape (36, 12)

ALL_COLS = FEATURE_COLS + ["glucose"]   # 12 columns total, glucose last

HISTORY_WINDOW_MINUTES = TIME_STEPS * 5  # 180 min — how far back we look

# ── Load model and scalers once at startup ────────────────────────────

MODEL_DIR = Path("/")   # adjust to your actual path

@app.on_event("startup")
def load_artifacts():
    global model, scaler_X, scaler_y
    model    = keras.models.load_model(MODEL_DIR / "lstm_model.h5")
    scaler_y = joblib.load(MODEL_DIR / "scaler_y.pkl")
    scaler_X = joblib.load(MODEL_DIR / "scaler_X.pkl")
    print("Scaler X ready:", scaler_X)
    print("Scaler y ready:", scaler_y)
    print(f"Model input shape: {model.input_shape}")   # should be (None, 36, 12)


# ── Pydantic request models ───────────────────────────────────────────

class MealLog(BaseModel):
    carbs: float
    meal_type: str
    logged_at: datetime

class BolusLog(BaseModel):
    dose_units: float
    logged_at: datetime

class ActivityLog(BaseModel):
    steps: float
    logged_at: datetime

class CGMReading(BaseModel):
    glucose: float
    timestamp: datetime

class LogEntryPayload(BaseModel):
    meals: List[MealLog]
    boluses: List[BolusLog]
    activity: List[ActivityLog]
    cgm_preview: List[CGMReading]   # must cover at least 3 h (36 readings)


# ── Log-history slice: grab the last TIME_STEPS-worth of relevant data ──

def slice_recent_log(payload: LogEntryPayload) -> LogEntryPayload:
    """
    Given a full log-history payload, return a new payload that contains only
    the entries falling within the last HISTORY_WINDOW_MINUTES (180 min) of
    CGM data.  This lets callers pass their entire log without pre-filtering.

    The cutoff is derived from the latest CGM timestamp, so the window is
    always anchored to the most recent glucose reading rather than wall-clock
    time (safe for replaying historical data too).
    """
    if not payload.cgm_preview:
        return payload

    latest_cgm = max(r.timestamp for r in payload.cgm_preview)

    # Make latest_cgm timezone-aware if needed, to compare with other fields
    def _ensure_tz(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    latest_cgm = _ensure_tz(latest_cgm)
    cutoff = latest_cgm - pd.Timedelta(minutes=HISTORY_WINDOW_MINUTES)

    def _within(dt: datetime) -> bool:
        return _ensure_tz(dt) >= cutoff

    return LogEntryPayload(
        cgm_preview=[r for r in payload.cgm_preview if _within(r.timestamp)],
        meals=[m for m in payload.meals if _within(m.logged_at)],
        boluses=[b for b in payload.boluses if _within(b.logged_at)],
        activity=[a for a in payload.activity if _within(a.logged_at)],
    )


# ── Feature engineering (mirrors training pipeline) ───────────────────

STEPS_WINDOW    = 10
MEAL_LAMBDA     = 1 / 60
MEAL_MAX_MIN    = 240
INSULIN_LAMBDA  = 0.02
INSULIN_MAX_MIN = 300


def build_feature_df(payload: LogEntryPayload) -> pd.DataFrame:
    # 1. Build 5-min grid anchored to CGM readings
    times = sorted([r.timestamp for r in payload.cgm_preview])
    start = pd.Timestamp(times[0]).floor(STEP_FREQ)
    end   = pd.Timestamp(times[-1]).floor(STEP_FREQ)
    grid  = pd.date_range(start=start, end=end, freq=STEP_FREQ)

    df = pd.DataFrame(index=grid)

    # 2. CGM glucose
    cgm_series = pd.Series(
        data=[r.glucose for r in payload.cgm_preview],
        index=[pd.Timestamp(r.timestamp).floor(STEP_FREQ) for r in payload.cgm_preview],
        dtype=float,
    )
    cgm_series = cgm_series.groupby(level=0).mean()
    df["glucose"] = cgm_series.reindex(grid)

    if df["glucose"].isna().any():
        missing = df[df["glucose"].isna()]
        print(missing)
        raise ValueError("Missing CGM readings after 5-minute alignment.")

    df["glucose"] = df["glucose"].interpolate(method="linear", limit=INTERP_LIMIT)

    # 3. bolus_raw
    df["bolus_raw"] = 0.0
    for b in payload.boluses:
        idx = grid.get_indexer([pd.Timestamp(b.logged_at)], method="nearest")[0]
        if 0 <= idx < len(grid):
            df.iloc[idx, df.columns.get_loc("bolus_raw")] += b.dose_units

    # 4. carbs + meal type one-hots
    for col in ["carbs", "meal_Breakfast", "meal_Dinner",
                "meal_HypoCorrection", "meal_Lunch", "meal_Snack"]:
        df[col] = 0.0

    type_col = {
        "Breakfast":      "meal_Breakfast",
        "Dinner":         "meal_Dinner",
        "HypoCorrection": "meal_HypoCorrection",
        "Lunch":          "meal_Lunch",
        "Snack":          "meal_Snack",
    }
    for meal in payload.meals:
        idx = grid.get_indexer([pd.Timestamp(meal.logged_at)], method="nearest")[0]
        if 0 <= idx < len(grid):
            df.iloc[idx, df.columns.get_loc("carbs")] += meal.carbs
            col = type_col.get(meal.meal_type)
            if col:
                df.iloc[idx, df.columns.get_loc(col)] = 1.0

    # 5. steps
    df["steps"] = 0.0
    for a in payload.activity:
        idx = grid.get_indexer([pd.Timestamp(a.logged_at)], method="nearest")[0]
        if 0 <= idx < len(grid):
            df.iloc[idx, df.columns.get_loc("steps")] += a.steps

    # 6. Derived features
    df["steps_weighted_avg"] = _steps_weighted_avg(df["steps"].values)
    df["meal_activity"]      = _meal_activity(grid, df["carbs"].values)
    df["insulin_activity"]   = _insulin_activity(grid, df["bolus_raw"].values)

    return df[ALL_COLS]   # enforce column order


def _steps_weighted_avg(steps: np.ndarray, window: int = STEPS_WINDOW) -> np.ndarray:
    weights = np.arange(window, 0, -1)
    out = np.zeros(len(steps))
    for i in range(len(steps)):
        start = max(0, i - window + 1)
        w = steps[start:i + 1]
        out[i] = np.dot(w, weights[-len(w):]) / window
    return out


def _meal_activity(grid: pd.DatetimeIndex, carbs: np.ndarray) -> np.ndarray:
    out = np.zeros(len(grid))
    for i in np.where(carbs > 0)[0]:
        elapsed = (grid - grid[i]) / pd.Timedelta(minutes=1)
        mask = (elapsed >= 0) & (elapsed <= MEAL_MAX_MIN)
        t = elapsed[mask]
        out[mask] += carbs[i] * MEAL_LAMBDA * t * np.exp(-MEAL_LAMBDA * t)
    return out


def _insulin_activity(grid: pd.DatetimeIndex, bolus_raw: np.ndarray) -> np.ndarray:
    out = np.zeros(len(grid))
    for i in np.where(bolus_raw > 0)[0]:
        elapsed = (grid - grid[i]) / pd.Timedelta(minutes=1)
        mask = (elapsed >= 0) & (elapsed <= INSULIN_MAX_MIN)
        t = elapsed[mask]
        out[mask] += bolus_raw[i] * INSULIN_LAMBDA * t * np.exp(-INSULIN_LAMBDA * t)
    return out


# ── Scaling + sequence building (mirrors training loop) ───────────────

def prepare_lstm_input(df: pd.DataFrame, scaler_X, scaler_y) -> np.ndarray:
    """
    Replicates exactly what training does before create_sequences():

        X_train_scaled = scaler_X.fit_transform(train_df[features])   # 11 cols
        y_scaled       = scaler_y.fit_transform(train_df[['glucose']]) # 1 col
        combined       = np.hstack([X_scaled, y_scaled])               # 12 cols
        X, y = create_sequences(combined, y_scaled, time_steps=36)
    """
    feature_vals = df[FEATURE_COLS].values          # (T, 11)
    glucose_vals = df[["glucose"]].values            # (T, 1)

    X_scaled = scaler_X.transform(feature_vals)      # (T, 11)
    y_scaled = scaler_y.transform(glucose_vals)      # (T, 1)

    combined = np.hstack([X_scaled, y_scaled])       # (T, 12)

    if len(combined) < TIME_STEPS:
        raise ValueError(
            f"Not enough history: need {TIME_STEPS} steps (3 h), "
            f"got {len(combined)}. Send more CGM readings."
        )

    window = combined[-TIME_STEPS:]                  # (36, 12)

    if np.isnan(window).any():
        raise ValueError(
            "NaN values remain in the input window after interpolation. "
            "Check for CGM gaps longer than 10 minutes."
        )

    return window[np.newaxis, ...]                   # (1, 36, 12)


# ── Endpoint ──────────────────────────────────────────────────────────

@app.post("/api/predict")
async def predict(payload: LogEntryPayload):
    # Trim the full log-history down to the relevant 3-hour window
    payload = slice_recent_log(payload)

    if len(payload.cgm_preview) < TIME_STEPS:
        raise HTTPException(
            status_code=400,
            detail=(f"Need at least {TIME_STEPS} CGM readings (3 hours). "
                f"Received {len(payload.cgm_preview)}."
            ),
        )

    try:
        df = build_feature_df(payload)
        X  = prepare_lstm_input(df, scaler_X, scaler_y)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    y_pred_scaled = model.predict(X)                            # (1, 1)
    y_pred_mgdl   = scaler_y.inverse_transform(y_pred_scaled)   # → mg/dL

    predicted_glucose = float(y_pred_mgdl[0, 0])
    prediction_time   = df.index[-1] + pd.Timedelta(minutes=HORIZON * 5)

    return {
        "predicted_glucose_mgdl": round(predicted_glucose, 1),
        "prediction_horizon_min": HORIZON * 5,          # 30 minutes
        "prediction_for_time":    prediction_time.isoformat(),
        "current_glucose_mgdl":   float(df["glucose"].iloc[-1]),
        "input_steps_used":       TIME_STEPS,
    }