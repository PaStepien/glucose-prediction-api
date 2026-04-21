import base64
from http.client import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from const.text_body import TextRequest
from model.tts.tts import generate_audio
from const.question_body import QuestionRequest
from model.mock_data import mock_data
from model.assistant_service import handle_user_message
from glucose_prediction_service import GlucosePredictionService
from const.time_step import GlucosePredictionInput
from fastapi import FastAPI, Response


app = FastAPI(title = "Glucose Level Prediction API", version = "1.0")
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
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
) 

@app.get("/")
def home():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict_glucose_level(request: GlucosePredictionInput):
    
    print(f"Received time steps: {request.time_steps}")
    if len(request.time_steps) != 36:
        return {"error": "Exactly 36 time steps are required."}
    
    input_data = service.prepare_input(request.time_steps)
    prediction = service.model.predict(input_data, verbose=0)
    predicted_glucose = service.scaler_y.inverse_transform(prediction)[0][0]
    print(f"Predicted glucose: {predicted_glucose}")
    return {"predicted_glucose": float(predicted_glucose)}
     
@app.post("/ask")
def ask_question(request: QuestionRequest):
    response = handle_user_message(
        message=request.question,
        data=mock_data  
    )
    
    try:
        audio_bytes = generate_audio(response["answer"])  # use answer for TTS
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