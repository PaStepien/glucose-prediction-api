import joblib
from tensorflow import keras
import numpy as np
import pandas as pd

class GlucosePredictionService:
    def __init__(self):
        # Load the pre-trained model
        self.model = keras.models.load_model('lstm_model.h5')
        self.scaler_x = joblib.load('scaler_X.pkl')
        self.scaler_y = joblib.load('scaler_Y.pkl')
        
    
    def prepare_input(self, request_data):
        
        
        df = pd.DataFrame([step.dict() for step in request_data])
         # 2. Identify the features (Everything except ID, Time, and Glucose)
        # This list MUST match the column order of your original training CSV
        
        feature_cols = [
        'bolus_raw', 'insulin_activity', 'carbs', 'meal_Breakfast', 
        'meal_Dinner', 'meal_HypoCorrection', 'meal_Lunch', 
        'meal_Snack', 'meal_activity', 'steps', 'steps_weighted_avg'
    ]
    
        # 3. Scale them using your saved .pkl files
        X_scaled = self.scaler_x.transform(df[feature_cols])
        y_scaled = self.scaler_y.transform(df[['glucose']])
        
        # 4. Combine them (Horizontal Stack)
        # This creates a (36, 12) matrix: 11 features + 1 glucose
        combined = np.hstack([X_scaled, y_scaled])
        
        # 5. Reshape for the LSTM: (Batch, TimeSteps, Features)
        # Result: (1, 36, 12)
        return combined.reshape(1, 36, 12)
    
        