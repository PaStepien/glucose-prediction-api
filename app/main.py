import base64
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
import os

from const.text_body import TextRequest
from const.question_body import QuestionRequest
from const.time_step import GlucosePredictionInput

from model.tts.tts import generate_audio
from model.mock_data import mock_data
from model.assistant_service import handle_user_message
from glucose_prediction_service import GlucosePredictionService
from formData.processData import build_feature_df, prepare_lstm_input
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env") 


app = FastAPI(title="Glucose Level Prediction API", version="1.0")

service = GlucosePredictionService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:8081",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None


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

class LogEntryRequest(BaseModel):
    user_id: str
    entry_date: datetime
    meals: List[MealLog]
    boluses: List[BolusLog]
    activity: List[ActivityLog]
    cgm_preview: List[CGMReading]  
                                  

class PredictionResponse(BaseModel):
    predicted_glucose_mgdl: float
    prediction_horizon_min: int
    prediction_for_time: str
    current_glucose_mgdl: float
    input_steps_used: int


lstm_model = None
scaler_X   = None
scaler_y   = None

@app.on_event("startup")
def load_model_artifacts():
    global lstm_model, scaler_X, scaler_y
    from pathlib import Path
    import joblib, keras

    artifact_dir = Path(".")
    try:
        lstm_model = keras.models.load_model(artifact_dir / "lstm_model.h5")
        scaler_X   = joblib.load(artifact_dir / "scaler_X.pkl")
        scaler_y   = joblib.load(artifact_dir / "scaler_y.pkl")
        print("✅ Model + scalers loaded")
    except Exception as e:
        print(f"⚠️ Failed loading artifacts: {e}")

# ── BASIC ENDPOINTS ───────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Hello World"}

@app.get("/health")
def health():
    
    print(f"Supabase URL: {SUPABASE_URL}")
    print(f"Supabase Key: {'set' if SUPABASE_KEY else 'not set'}")
    return {
        "status":       "ok",
        "time":         datetime.now(timezone.utc).isoformat(),
        "model_loaded": lstm_model is not None,
        "db_connected": supabase is not None,
    }

# ── MOCK CGM ──────────────────────────────────────────────────────────
# Called on FORM LOAD so the CGM preview card shows values immediately.
# Frontend stores the returned readings in state and sends them
# as cgm_preview when the user presses Save.

@app.get("/api/cgm/mock/{user_id}")
def mock_cgm(user_id: str, n: int = Query(default=36, ge=36, le=288)):
    now   = datetime.now(timezone.utc)
    times = [now - timedelta(minutes=5 * i) for i in range(n - 1, -1, -1)]

    base   = 90.0
    noise  = np.random.normal(0, 1.5, n)
    peak   = np.concatenate([
        np.linspace(0, 20, n // 2),
        np.linspace(20, 0, n - n // 2),
    ])
    values = [round(float(base + peak[i] + noise[i]), 1) for i in range(n)]

    return {
        "user_id": user_id,
        "readings": [
            {"glucose": values[i], "timestamp": times[i].isoformat()}
            for i in range(n)
        ],
    }

# ── LOG ENTRY ─────────────────────────────────────────────────────────
# Triggered when user presses Save on the form.
# Saves raw inputs only — no feature engineering here.
# Returns the new row id so the frontend can pass it along if needed.

@app.post("/api/log-entry")
def save_log_entry(payload: LogEntryRequest):
    row = {
        "user_id": payload.user_id,
        "entry_date": payload.entry_date.isoformat(),
        "meals": [m.model_dump(mode="json") for m in payload.meals],
        "boluses": [b.model_dump(mode="json") for b in payload.boluses],
        "activity": [a.model_dump(mode="json") for a in payload.activity],
        "cgm": [c.model_dump(mode="json") for c in payload.cgm_preview],
    }

    if supabase is None:
        return {"status": "saved (no db)", "id": None}

    # ── 1. Save RAW ───────────────────────────────────────────
    result = supabase.table("log_entries").insert(row).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to save log entry")

    saved = result.data[0]
    log_id = saved["id"]

    print(f"✅ Saved log entry id={log_id}")

    # ── 2. PROCESS + PREDICT ──────────────────────────────────
    try:
        if lstm_model is None:
            raise Exception("Model not loaded")

        df = build_feature_df(payload)
        X = prepare_lstm_input(df, scaler_X, scaler_y)

        y_pred_scaled = lstm_model.predict(X)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        predicted_glucose = float(y_pred[0, 0])
        current_glucose = float(df["glucose"].iloc[-1])
        prediction_time = df.index[-1] + timedelta(minutes=30)

        latest = df.iloc[-1]

        # ── 3. SAVE PROCESSED ──────────────────────────────────
        processed_row = {
            "user_id": payload.user_id,
            "log_entry_id": log_id,
            "prediction_for_time": prediction_time.isoformat(),

            "predicted_glucose_mgdl": predicted_glucose,
            "current_glucose_mgdl": current_glucose,
            "horizon_minutes": 30,

            # features (aligned!)
            "glucose": float(latest["glucose"]),
            "bolus_raw": float(latest["bolus_raw"]),
            "insulin_activity": float(latest["insulin_activity"]),
            "carbs": float(latest["carbs"]),
            "meal_breakfast": float(latest["meal_Breakfast"]),
            "meal_dinner": float(latest["meal_Dinner"]),
            "meal_hypocorrection": float(latest["meal_HypoCorrection"]),
            "meal_lunch": float(latest["meal_Lunch"]),
            "meal_snack": float(latest["meal_Snack"]),
            "meal_activity": float(latest["meal_activity"]),
            "steps": float(latest["steps"]),
            "steps_weighted_avg": float(latest["steps_weighted_avg"]),

            # optional debug
            "lstm_window": df.tail(36).to_dict(orient="records"),
        }

        pred_result = supabase.table("processed_predictions").insert(processed_row).execute()

        print(processed_row)

        prediction_id = pred_result.data[0]["id"] if pred_result.data else None

    except Exception as e:
        print(f"⚠️ Processing failed: {e}")
        prediction_id = None
        predicted_glucose = None

    # ── RESPONSE ──────────────────────────────────────────────
    return {
        "status": "saved",
        "log_entry_id": log_id,
        "prediction_id": prediction_id,
        "predicted_glucose_mgdl": predicted_glucose,
    }

@app.get("/api/log-history/{user_id}")
def get_log_history(
    user_id: str,
    days: int = Query(default=1, ge=1, le=30),
):
    if supabase is None:
        return []

    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    result = (
        supabase.table("processed_predictions")
        .select("*")
        .eq("user_id", user_id)
        .gte("created_at", since)
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )
    return result.data or []

@app.post("/api/predict", response_model=PredictionResponse)
def predict_api(payload: LogEntryRequest):
    if lstm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(payload.cgm_preview) < 36:
        raise HTTPException(
            status_code=400,
            detail="Need at least 36 CGM readings (3 hours). "
                   "Make sure cgm_preview is populated from /api/cgm/mock.",
        )

    try:
        df = build_feature_df(payload)
        X  = prepare_lstm_input(df, scaler_X, scaler_y)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    y_pred_scaled     = lstm_model.predict(X)
    y_pred            = scaler_y.inverse_transform(y_pred_scaled)
    predicted_glucose = round(float(y_pred[0, 0]), 1)
    current_glucose   = float(df["glucose"].iloc[-1])
    prediction_time   = df.index[-1] + timedelta(minutes=30)

    return {
        "predicted_glucose_mgdl": predicted_glucose,
        "prediction_horizon_min": 30,
        "prediction_for_time":    prediction_time.isoformat(),
        "current_glucose_mgdl":   current_glucose,
        "input_steps_used":       36,
    }


@app.post("/predict")
async def predict_glucose_level(request: GlucosePredictionInput):
    if len(request.time_steps) != 36:
        return {"error": "Exactly 36 time steps are required."}

    input_data        = service.prepare_input(request.time_steps)
    prediction        = service.model.predict(input_data, verbose=0)
    predicted_glucose = service.scaler_y.inverse_transform(prediction)[0][0]

    return {"predicted_glucose": float(predicted_glucose)}




@app.post("/ask")
def ask_question(request: QuestionRequest):
    response = handle_user_message(
        message=request.question,
        data=mock_data,
    )

    try:
        audio_bytes = generate_audio(response["answer"])
        audio_b64   = base64.b64encode(audio_bytes).decode("utf-8")

        return JSONResponse(content={
            "answer":            response["answer"],
            "predicted_glucose": response["predicted_glucose"],
            "intent":            response["intent"],
            "audio":             f"data:audio/wav;base64,{audio_b64}",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/talk")
def talk(request: TextRequest):
    try:
        audio_bytes = generate_audio(request.text)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))