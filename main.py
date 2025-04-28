# Modified main.py for FastAPI with Supabase integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from supabase import create_client
import os
import io
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("SUPABASE_URL or SUPABASE_KEY environment variables are not set!")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Global model cache
model = None
numerical_imputer = None
categorical_imputer = None
scaler = None
label_encoder = None
label_encoder_y = None
training_features = None

class CropInput(BaseModel):
    latitude: float
    longitude: float
    ph: float
    nitrogen: float
    phosphorus: float
    potassium: float
    organic_carbon: float
    clay_content: float
    rainfall: float
    temperature: float
    humidity: float
    zone: str

def load_component(component_name: str):
    try:
        logger.info(f"Loading {component_name} from Supabase Storage...")
        response = supabase.storage.from_("models").download(f"{component_name}.pkl")
        
        if response is None:
            raise Exception(f"{component_name} download failed")
            
        return joblib.load(io.BytesIO(response))
    except Exception as e:
        logger.error(f"Error loading {component_name}: {e}")
        raise

def load_model_system():
    global model, numerical_imputer, categorical_imputer, scaler, label_encoder, label_encoder_y, training_features
    
    components = [
        ('model', 'crop_recommendation_model'),
        ('numerical_imputer', 'numerical_imputer'),
        ('categorical_imputer', 'categorical_imputer'),
        ('scaler', 'scaler'),
        ('label_encoder', 'label_encoder'),
        ('label_encoder_y', 'label_encoder_y'),
        ('training_features', 'training_features')
    ]
    
    for var_name, component_name in components:
        try:
            globals()[var_name] = load_component(component_name)
            logger.info(f"Successfully loaded {component_name}")
        except Exception as e:
            logger.error(f"Critical error loading {component_name}: {e}")
            raise

@app.on_event("startup")
async def startup_event():
    load_model_system()

def preprocess_input(input_data: dict) -> pd.DataFrame:
    try:
        # Create DataFrame from input
        new_data = pd.DataFrame([input_data.dict()])
        
        # Handle missing values
        num_cols = new_data.select_dtypes(include=[np.number]).columns
        cat_cols = new_data.select_dtypes(include=[object]).columns
        
        new_data[num_cols] = numerical_imputer.transform(new_data[num_cols])
        new_data[cat_cols] = categorical_imputer.transform(new_data[cat_cols])
        
        # Feature engineering
        new_data['npk_ratio'] = (new_data['nitrogen'] + new_data['phosphorus'] + new_data['potassium']) / 3
        new_data['gdd'] = (new_data['temperature'] - 10).clip(lower=0)
        
        # Encode categorical features
        new_data['zone'] = label_encoder.transform(new_data['zone'])
        
        # Ensure all training features are present
        for feature in training_features:
            if feature not in new_data.columns:
                new_data[feature] = 0  # Add missing features with default
        
        # Reorder columns to match training order
        new_data = new_data.reindex(columns=training_features)
        
        # Scale numerical features
        num_cols = [col for col in num_cols if col in training_features]
        new_data[num_cols] = scaler.transform(new_data[num_cols])
        
        return new_data
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

GROWING_TIPS = {
    "Maize": {
        "Soil": "Loamy well-drained soil, rich in organic matter.",
        "Watering": "Requires consistent moisture, especially during flowering.",
        "Harvest": "Harvest when husks turn brown and kernels are fully formed."
    },
    "Beans": {
        "Soil": "Well-drained sandy loam soil with moderate fertility.",
        "Watering": "Water moderately, avoid waterlogging.",
        "Harvest": "Pick pods when they are firm and before seeds bulge."
    },
    "Rice": {
        "Soil": "Clayey or silty soil that retains water well.",
        "Watering": "Needs standing water during most of the growing season.",
        "Harvest": "Harvest when grains turn golden yellow."
    },
    "Wheat": {
        "Soil": "Well-drained loamy soil with neutral pH.",
        "Watering": "Moderate watering, critical at tillering and flowering stages.",
        "Harvest": "Harvest when plants are golden and grains are hard."
    },
    "Sorghum": {
        "Soil": "Sandy loam soils with good drainage.",
        "Watering": "Drought-tolerant, minimal watering needed.",
        "Harvest": "Harvest when grains are hard and dry."
    },
    "Millet": {
        "Soil": "Light sandy soils that drain quickly.",
        "Watering": "Requires less water compared to other cereals.",
        "Harvest": "Harvest when heads are dry and seeds harden."
    },
    "Tomato": {
        "Soil": "Loamy soil rich in organic content, well-draining.",
        "Watering": "Regular watering to maintain even soil moisture.",
        "Harvest": "Harvest when fruits are fully red and firm."
    },
    "Potato": {
        "Soil": "Loose, well-drained sandy loam rich in organic matter.",
        "Watering": "Keep soil moist during tuber formation.",
        "Harvest": "Harvest after foliage dies back."
    },
    "Cabbage": {
        "Soil": "Fertile, well-drained soils with high moisture retention.",
        "Watering": "Frequent watering for steady moisture supply.",
        "Harvest": "Harvest when heads are firm and fully formed."
    },
    "Spinach": {
        "Soil": "Rich, moist soil with good drainage.",
        "Watering": "Keep soil consistently moist.",
        "Harvest": "Harvest outer leaves when big enough to use."
    },
}

@app.post("/predict")
async def predict(input_data: CropInput):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[0]
        
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_crops = label_encoder_y.inverse_transform(top_indices)
        
        recommendations = []
        for crop, prob in zip(top_crops, probabilities[top_indices]):
            tips = GROWING_TIPS.get(crop, {
                "Soil": "General fertile soil.",
                "Watering": "Moderate watering.",
                "Harvest": "Harvest when mature."
            })
            recommendations.append({
                "name": crop,
                "confidence": round(prob * 100, 2),
                "growing_tips": tips
            })
        
        return {
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


