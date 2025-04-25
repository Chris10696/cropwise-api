from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from supabase import create_client
import os

app = FastAPI()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class CropInput(BaseModel):
    latitude: float
    longitude: float
    pH: float
    nitrogen: float
    phosphorus: float
    potassium: float
    organic_carbon: float
    clay_content: float
    rainfall: float
    temperature: float
    humidity: float

@app.post("/predict")
async def predict(input_data: CropInput):
    # Load model from Supabase Storage
    model_bytes = supabase.storage.from_("models").download("crop_model.pkl")
    model = joblib.load(model_bytes)

    input_array = np.array([[ 
        input_data.latitude,
        input_data.longitude,
        input_data.pH,
        input_data.nitrogen,
        input_data.phosphorus,
        input_data.potassium,
        input_data.organic_carbon,
        input_data.clay_content,
        input_data.rainfall,
        input_data.temperature,
        input_data.humidity
    ]])

    prediction = model.predict(input_array)
    probabilities = model.predict_proba(input_array)

    return {
        "recommended_crop": prediction[0],
        "confidence": float(np.max(probabilities)),
        "all_probabilities": probabilities.tolist()
    }
