import uuid
import pandas as pd
import numpy as np
import mlflow.xgboost
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the project root is in the path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.features.engineer import FeatureEngineer

# Global holders for our model and transformer
model = None
feature_engineer = None

class TransactionInput(BaseModel):
    """
    Schema for incoming transaction data.
    These are the raw features before any engineering.
    """
    TransactionAmt: float
    ProductCD: str
    card1: int
    card4: str
    TransactionDT: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for loading the model and transformer on startup.
    """
    global model, feature_engineer
    
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = "Fraud_Detection_IEEE_CIS"
    
    try:
        logger.info(f"Attempting to load the latest model from experiment: {experiment_name}")
        
        # 1. Retrieve the latest run from the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            raise RuntimeError(f"Experiment '{experiment_name}' not found.")
            
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            raise RuntimeError(f"No runs found in experiment '{experiment_name}'.")
            
        run_id = runs.iloc[0].run_id
        logger.info(f"Loading artifacts from run_id: {run_id}")
        
        # 2. Load the XGBoost model using MLflow's native loader
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.xgboost.load_model(model_uri)
        logger.info("XGBoost model loaded successfully.")
        
        # 3. Load the fitted FeatureEngineer artifact
        # We download the artifact from MLflow tracking store
        local_fe_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, 
            artifact_path="feature_engineer.pkl"
        )
        with open(local_fe_path, "rb") as f:
            feature_engineer = pickle.load(f)
        logger.info("FeatureEngineer transformer loaded successfully.")
        
    except Exception as e:
        logger.warning(f"Failed to load artifacts from MLflow: {e}")
        logger.info("Checking for local fallback files (model.pkl / feature_engineer.pkl)...")
        
        # Fallback to local files if MLflow is unavailable
        try:
            # Note: We prioritize the latest local files if they exist
            if Path("feature_engineer.pkl").exists():
                with open("feature_engineer.pkl", "rb") as f:
                    feature_engineer = pickle.load(f)
                logger.info("Loaded local feature_engineer.pkl")
            
            # For the model, XGBoost might be saved as a pickle or json
            # Here we assume a simple pickle fallback for the demonstration
            if Path("model.pkl").exists():
                with open("model.pkl", "rb") as f:
                    model = pickle.load(f)
                logger.info("Loaded local model.pkl")
        except Exception as fe:
            logger.error(f"Fallback loading failed: {fe}")
            
    if model is None or feature_engineer is None:
        logger.error("API started without a valid model or transformer. Predictions will fail.")
    
    yield
    # Cleanup logic (if any) goes here
    logger.info("Shutting down API...")

# Initialize FastAPI with our lifespan manager
app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    description="Real-time fraud detection service using XGBoost and MLflow.",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/predict")
async def predict(transaction: TransactionInput):
    """
    Predicts whether a transaction is fraudulent.
    """
    if model is None or feature_engineer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model or Feature Transformer is not initialized. Check server logs."
        )
    
    try:
        # 1. Convert input to a single-row DataFrame for the transformer
        input_df = pd.DataFrame([transaction.model_dump()])
        
        # 2. Apply the fitted FeatureEngineer transformations
        processed_df = feature_engineer.transform(input_df)
        
        # 3. Align with model features
        # XGBoost requires the exact same features in the same order as training.
        # We retrieve the feature names from the model's booster.
        try:
            model_features = model.get_booster().feature_names
        except AttributeError:
            # Fallback for some versions/wrappers
            model_features = getattr(model, "feature_names_in_", None)
            
        if model_features:
            # Create a full feature set initialized with the training default (-999)
            full_X = pd.DataFrame(index=[0], columns=model_features)
            full_X = full_X.fillna(-999)
            
            # Update the template with the features we have processed
            for col in processed_df.columns:
                if col in model_features:
                    full_X[col] = processed_df[col].values
            
            # Ensure the final DataFrame has the exact columns and order required
            X = full_X[model_features].astype(float)
        else:
            # Fallback if feature names aren't available
            X = processed_df.select_dtypes(include=[np.number])
            if 'TransactionID' in X.columns:
                X = X.drop(columns=['TransactionID'])
            
        # 4. Perform Inference
        probability = float(model.predict_proba(X)[0][1])
        
        # 5. Determine Risk Level
        risk_level = "HIGH" if probability > 0.7 else "LOW"
        
        return {
            "transaction_id": str(uuid.uuid4()),
            "fraud_probability": round(probability, 4),
            "risk_level": risk_level,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    """
    return {
        "status": "online",
        "model_ready": model is not None,
        "transformer_ready": feature_engineer is not None
    }
