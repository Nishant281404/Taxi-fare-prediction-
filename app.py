from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('fare_prediction_model.pkl')

# Define the input data model
class FarePredictionInput(BaseModel):
    trip_distance: float
    passenger_count: int
    pickup_hour: int
    pickup_day: int
    pickup_month: int
    VendorID: int
    RatecodeID: int
    PULocationID: int
    DOLocationID: int
    payment_type: int

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
def predict_fare(input_data: FarePredictionInput):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data.dict()])

    # Use the model to predict the fare
    prediction = model.predict(input_df)

    return {"predicted_fare": prediction[0]}

# To run the server, use the command:
# uvicorn app:app --reload

    