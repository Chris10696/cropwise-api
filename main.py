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
    pH: float
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

@app.post("/predict")
async def predict(input_data: CropInput):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
            
        # Preprocess input data
        processed_data = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)
        
        # Decode prediction
        pred_label = label_encoder_y.inverse_transform(prediction)
        
        return {
            "recommended_crop": pred_label[0],
            "confidence": float(np.max(probabilities)),
            "all_probabilities": probabilities.tolist()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
