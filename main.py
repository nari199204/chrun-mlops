# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)  # <- Make sure this is indented under the "with" line

app = FastAPI()

# Define the input schema
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def read_root():
    return {"message": "Churn prediction API is running 🎉"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    features = np.array([[value for value in data.dict().values()]])
    prediction = model.predict(features)[0]
    return {"churn": bool(prediction)}
