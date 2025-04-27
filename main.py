from fastapi import FastAPI
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
    raise Exception("SUPABASE_URL or SUPABASE_KEY environment variables are not set!")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Global model cache
model = None

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

def load_model():
    global model
    try:
        print("Loading model from Supabase Storage...")
        response = supabase.storage.from_("models").download("crop_model.pkl")

        if response is None:
            raise Exception("Model download failed. Check if the file exists in Supabase Storage.")

        # Make sure to load the model from bytes
        model_file = io.BytesIO(response)
        model = joblib.load(model_file)
        print("Model loaded successfully!")

    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/predict")
async def predict(input_data: CropInput):
    try:
        if model is None:
            raise Exception("Model is not loaded yet.")

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
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}
