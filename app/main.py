from glucose_prediction_service import GlucosePredictionService
from time_step import GlucosePredictionInput
from fastapi import FastAPI

app = FastAPI(title = "Glucose Level Prediction API", version = "1.0")
service = GlucosePredictionService()

@app.get("/")
def home():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict_glucose_level(request: GlucosePredictionInput):
    
    if len(request.time_steps) != 36:
        return {"error": "Exactly 36 time steps are required."}
    
    input_data = service.prepare_input(request.time_steps)
    prediction = service.model.predict(input_data, verbose=0)
    predicted_glucose = service.scaler_y.inverse_transform(prediction)[0][0]
    return {"predicted_glucose": float(predicted_glucose)}
     
