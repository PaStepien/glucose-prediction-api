import json

from formData.processData import prepare_lstm_input
from model.helpers import build_assistant_context
from typing import Optional

from model.prompts.classifier_prompt import build_classifier_prompt
from model.client import query_llama, query_llama_classifier
from model.prompts.prompt import build_prompt


PREDICTION_API_URL = "http://localhost:8000/predict"

def detect_intent(message: str) -> str:
    prompt = build_classifier_prompt(message)
    return query_llama_classifier(prompt)

def handle_user_message(
    message: str,
    data: Optional,
    lstm_model,
    scaler_X,
    scaler_y
) -> dict:

    intent_response = detect_intent(message)
    print(f"Raw intent response: {intent_response}")
    
    try:
        intent_data = json.loads(intent_response)
        intent =  intent_data["intent"].strip().lower()
    except Exception as e:
        print(f"Error parsing intent: {e}")
        intent = "general"
        
    print(f"Detected intent: {intent}")

    predicted_glucose: Optional[float] = None
    
    context = build_assistant_context(data)
        
    print(f"Built context: {context}")

    if intent == "general":
        prompt = message
    else:
        if data is None or (not hasattr(data, 'empty') and not isinstance(data, dict)):
            return {"error": "Data must be a DataFrame or dictionary for prediction."}
        
        if intent == "predict":
            if lstm_model is None or scaler_X is None or scaler_y is None:
                return {"error": "Model/scalers not provided for prediction."}
            if not isinstance(data, dict):
                return {"error": "Data must be a dictionary for prediction."}
            X = prepare_lstm_input(data, scaler_X, scaler_y)
            y_pred_scaled = lstm_model.predict(X)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            predicted_glucose = float(y_pred[0, 0])

            print(f"Predicted glucose: {predicted_glucose}")
    
        
        prompt = build_prompt(
            question=message,
            context=context,
            predicted_glucose=predicted_glucose
        )

    answer = query_llama(prompt)

    return {
        "answer": answer,
        "predicted_glucose": predicted_glucose,
        "intent": intent
    }

   