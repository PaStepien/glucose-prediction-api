import requests
from typing import Optional

from client import query_llama
from prompt import build_prompt


PREDICTION_API_URL = "http://localhost:8000/predict"

def detect_intent(message: str) -> str:
    msg = message.lower()

    prediction_keywords = [
        "predict", "forecast", "future", "next",
        "going to be", "estimate", "will my glucose"
    ]

    explanation_keywords = [
        "why", "explain", "reason", "how come"
    ]

    if any(k in msg for k in prediction_keywords):
        return "prediction"

    if any(k in msg for k in explanation_keywords):
        return "explanation"

    return "general"


def call_prediction_api(time_steps: list) -> float:
    response = requests.post(
        PREDICTION_API_URL,
        json={"time_steps": time_steps}
    )

    if response.status_code != 200:
        raise Exception("Prediction service failed")

    return response.json()["predicted_glucose"]

def handle_user_message(
    message: str,
    data: dict
) -> dict:

    intent = detect_intent(message)

    predicted_glucose: Optional[float] = None

    if intent == "prediction":
        if "time_steps" not in data or len(data["time_steps"]) != 36:
            return {
                "error": "Prediction requires exactly 36 time steps."
            }

        predicted_glucose = call_prediction_api(data["time_steps"])

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