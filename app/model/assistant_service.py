from flask import json
import requests
from typing import Optional

from model.prompts.classifier_prompt import build_classifier_prompt
from model.client import query_llama, query_llama_classifier
from model.prompts.prompt import build_prompt


PREDICTION_API_URL = "http://localhost:8000/predict"

def detect_intent(message: str) -> str:
    prompt = build_classifier_prompt(message)
    return query_llama_classifier(prompt)


def call_prediction_api(time_steps: list) -> float:
    try:
        response = requests.post(
            PREDICTION_API_URL,
            json={"time_steps": time_steps},
            timeout=20,
        )
    except requests.RequestException as exc:
        raise Exception(f"Prediction service request failed: {exc}") from exc
    
    print(f"Prediction API response: {response.status_code} - {response.text}")

    if response.status_code != 200:
        raise Exception("Prediction service failed")

    return response.json()["predicted_glucose"]

def handle_user_message(
    message: str,
    data: dict
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

    if intent == "general":
        prompt = message
    else:
        if intent == "predict":
            if "time_steps" not in data or len(data["time_steps"]) != 36:
                return {
                    "error": "Prediction requires exactly 36 time steps."
                }

            predicted_glucose = call_prediction_api(data["time_steps"])
            print(f"Predicted glucose: {predicted_glucose}")
        
        prompt = build_prompt(
            question=message,
            data=data,
            predicted_glucose=predicted_glucose
        )

    answer = query_llama(prompt)

    return {
        "answer": answer,
        "predicted_glucose": predicted_glucose,
        "intent": intent
    }

   