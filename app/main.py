import base64
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── YOUR EXISTING IMPORTS ─────────────────────────────────────────────
from const.text_body import TextRequest
from const.question_body import QuestionRequest
from const.time_step import GlucosePredictionInput

from model.tts.tts import generate_audio
from model.mock_data import mock_data
from model.assistant_service import handle_user_message
from glucose_prediction_service import GlucosePredictionService
from formData.processData import build_feature_df, prepare_lstm_input

# ── APP ───────────────────────────────────────────────────────────────

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

# ── SIMPLE IN-MEMORY STORAGE (TEMP DB) ────────────────────────────────

fake_db = []

# ── SCHEMAS ──────────────────────────────────────────────────────────

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
    log_entry_id: Optional[int] = None

# ── LOAD MODEL + SCALERS ON STARTUP ───────────────────────────────────

lstm_model = None
scaler_X = None
scaler_y = None

@app.on_event("startup")
def load_model_artifacts():
    global lstm_model, scaler_X, scaler_y
    from pathlib import Path
    import joblib, keras

    artifact_dir = Path("model_artifacts")

    if not artifact_dir.exists():
        print("⚠️ model_artifacts/ not found — /api/predict will fail.")
        return

    try:
        lstm_model = keras.models.load_model(artifact_dir / "lstm_model.keras")
        scaler_X = joblib.load(artifact_dir / "scaler_X.pkl")
        scaler_y = joblib.load(artifact_dir / "scaler_y.pkl")
        print("✅ Model + scalers loaded")
    except Exception as e:
        print(f"⚠️ Failed loading artifacts: {e}")

# ── BASIC ENDPOINTS ───────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Hello World"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "model_loaded": lstm_model is not None,
    }

# ── ORIGINAL SIMPLE PREDICT (KEEPED) ──────────────────────────────────

@app.post("/predict")
async def predict_glucose_level(request: GlucosePredictionInput):

    if len(request.time_steps) != 36:
        return {"error": "Exactly 36 time steps are required."}

    input_data = service.prepare_input(request.time_steps)
    prediction = service.model.predict(input_data, verbose=0)
    predicted_glucose = service.scaler_y.inverse_transform(prediction)[0][0]

    return {"predicted_glucose": float(predicted_glucose)}

# ── NEW FULL PIPELINE PREDICT ─────────────────────────────────────────

@app.post("/api/predict", response_model=PredictionResponse)
def predict_api(payload: LogEntryRequest):

    if lstm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(payload.cgm_preview) < 36:
        raise HTTPException(
            status_code=400,
            detail="Need at least 36 CGM readings (3 hours)"
        )

    try:
        df = build_feature_df(payload)
        X = prepare_lstm_input(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    y_pred_scaled = lstm_model.predict(X)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    predicted_glucose = float(y_pred[0, 0])
    current_glucose = float(df["glucose"].iloc[-1])
    prediction_time = df.index[-1] + timedelta(minutes=30)

    return {
        "predicted_glucose_mgdl": round(predicted_glucose, 1),
        "prediction_horizon_min": 30,
        "prediction_for_time": prediction_time.isoformat(),
        "current_glucose_mgdl": current_glucose,
        "input_steps_used": 36,
        "log_entry_id": None,
    }

# ── LOG ENTRY (TEMP DB) ───────────────────────────────────────────────

@app.post("/api/log-entry")
def save_log_entry(payload: LogEntryRequest):
    fake_db.append(payload)
    return {"status": "saved", "total_entries": len(fake_db)}

# ── LOG HISTORY ───────────────────────────────────────────────────────

@app.get("/api/log-history/{user_id}")
def get_log_history(user_id: str):
    return [entry for entry in fake_db if entry.user_id == user_id]

# ── MOCK CGM ──────────────────────────────────────────────────────────

@app.get("/api/cgm/mock/{user_id}")
def mock_cgm(user_id: str, n: int = Query(default=36, ge=36, le=288)):

    now = datetime.now(timezone.utc)
    times = [now - timedelta(minutes=5 * i) for i in range(n - 1, -1, -1)]

    base = 90.0
    noise = np.random.normal(0, 1.5, n)
    peak = np.concatenate([
        np.linspace(0, 20, n // 2),
        np.linspace(20, 0, n - n // 2)
    ])

    values = [round(float(base + peak[i] + noise[i]), 1) for i in range(n)]

    return {
        "user_id": user_id,
        "readings": [
            {"glucose": values[i], "timestamp": times[i].isoformat()}
            for i in range(n)
        ],
    }

# ── LLM + TTS ─────────────────────────────────────────────────────────

@app.post("/ask")
def ask_question(request: QuestionRequest):
    response = handle_user_message(
        message=request.question,
        data=mock_data
    )

    try:
        audio_bytes = generate_audio(response["answer"])
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return JSONResponse(content={
            "answer": response["answer"],
            "predicted_glucose": response["predicted_glucose"],
            "intent": response["intent"],
            "audio": f"data:audio/wav;base64,{audio_b64}"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/talk")
def talk(request: TextRequest):
    try:
        audio_bytes = generate_audio(request.text)
        return Response(content=audio_bytes, media_type='audio/wav')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))