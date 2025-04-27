from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from supabase import create_client
import os
import io

app = FastAPI()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Supabase URL and Key must be set in environment variables.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Input model
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

# Global variable to store the model in memory
model = None

# Function to load model (if not already loaded)
def load_model():
    global model
    if model is None:
        model_file = supabase.storage.from_("models").download("crop_model.pkl")
        model = joblib.load(io.BytesIO(model_file))

# Healthcheck endpoint
@app.get("/")
async def root():
    return {"message": "CropWise API is running!"}

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: CropInput):
    try:
        # Load model once (if not already loaded)
        load_model()

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

        # Make prediction
        prediction = model.predict(input_array)
        probabilities = model.predict_proba(input_array)

        return {
            "recommended_crop": prediction[0],
            "confidence": float(np.max(probabilities)),
            "all_probabilities": probabilities.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
