import joblib
import pandas as pd
from pathlib import Path
from microsoft_model.config.config import config

def load_pipeline():
    save_path = Path(__file__).parent / config.pipeline_save_file
    return joblib.load(save_path)

def make_prediction(input_data: dict) -> dict:
    pipeline = load_pipeline()
    
    df = pd.DataFrame([input_data])
    df = df[config.numerical_features]
    
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]
    
    return {
        "prediction": "Bullish" if prediction == 1 else "Not Bullish",
        "probability": round(float(probability), 4)
    }