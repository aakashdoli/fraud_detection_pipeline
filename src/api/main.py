from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API", version="1.0.0")

class Transaction(BaseModel):
    user_id: str
    amount: float
    location: str
    timestamp: str

@app.get("/")
async def root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
async def predict(transaction: Transaction):
    # Placeholder for prediction logic
    return {"is_fraud": False, "score": 0.01}
