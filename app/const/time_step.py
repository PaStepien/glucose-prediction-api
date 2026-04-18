from pydantic import BaseModel

class TimeStep(BaseModel):
    glucose: float
    bolus_raw: float = 0.0
    insulin_activity: float = 0.0
    carbs: float = 0.0
    meal_Breakfast: float = 0.0
    meal_Dinner: float = 0.0
    meal_HypoCorrection: float = 0.0
    meal_Lunch: float = 0.0
    meal_Snack: float = 0.0
    meal_activity: float = 0.0
    steps: float = 0.0
    steps_weighted_avg: float = 0.0

class GlucosePredictionInput(BaseModel):
    time_steps: list[TimeStep]
    