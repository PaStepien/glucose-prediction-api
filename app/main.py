from const.question_body import QuestionRequest
from model.mock_data import mock_data
from model.assistant_service import handle_user_message
from glucose_prediction_service import GlucosePredictionService
from const.time_step import GlucosePredictionInput
from fastapi import FastAPI

app = FastAPI(title = "Glucose Level Prediction API", version = "1.0")
service = GlucosePredictionService()

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
    
    return response