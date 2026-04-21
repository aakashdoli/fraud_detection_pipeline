import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from pathlib import Path
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to sys.path to allow importing from src
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.features.engineer import FeatureEngineer

def run_training_pipeline():
    """
    Complete training pipeline for the IEEE-CIS Fraud Detection project.
    """
    # 1. Load Data
    data_path = project_root / "data" / "raw" / "train_transaction.csv"
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}. Please check the file path.")
        return

    logger.info("Loading raw dataset...")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']

    # 2. Process Features
    logger.info("Initializing FeatureEngineer and applying fit_transform...")
    fe = FeatureEngineer()
    X_engineered = fe.fit_transform(X)
    
    # Filter for numerical features and drop IDs for training
    # fit_transform handles encoding and returns a dataframe with some float/int columns
    X_model = X_engineered.select_dtypes(include=[np.number])
    if 'TransactionID' in X_model.columns:
        X_model = X_model.drop(columns=['TransactionID'])
    
    logger.info(f"Feature engineering complete. Data shape: {X_model.shape}")

    # 3. Train/Test Split (80/20 Stratified)
    logger.info("Splitting data into training and testing sets (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. MLflow Setup
    # Setting tracking URI to a local directory
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Fraud_Detection_IEEE_CIS")

    # 5. Train XGBoost
    # Calculate scale_pos_weight to handle the ~3.5% class imbalance
    # Formula: sum(negative_instances) / sum(positive_instances)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")

    with mlflow.start_run(run_name="XGBoost_Baseline"):
        logger.info("Training XGBClassifier...")
        
        xgb_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }
        
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)

        # 6. Evaluation and Logging
        logger.info("Generating predictions and calculating metrics...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        # Log parameters and metrics to MLflow
        mlflow.log_params(xgb_params)
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
            logger.info(f"Metric - {metric_name}: {value:.4f}")

        # Log the model artifact
        logger.info("Logging XGBoost model to MLflow...")
        mlflow.xgboost.log_model(model, artifact_path="model")
        
        # Save and log the FeatureEngineer object
        fe_path = "feature_engineer.pkl"
        with open(fe_path, "wb") as f:
            import pickle
            pickle.dump(fe, f)
        mlflow.log_artifact(fe_path)
        logger.info(f"FeatureEngineer artifact logged.")
        
        logger.info("Training pipeline finished successfully.")

if __name__ == "__main__":
    run_training_pipeline()
