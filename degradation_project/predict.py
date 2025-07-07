# degradation_project/predict.py

import os
import pandas as pd
import joblib

# Load paths
MODEL_PATH = os.path.join("degradation_project", "models", "degradation_model.pkl")
PREPROCESSOR_PATH = os.path.join("degradation_project", "models", "preprocessor.pkl")

# Load model and preprocessor once
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

def predict_degradation(input_dict: dict) -> dict:
    """
    Predict degradation metrics based on input scaffold parameters.

    Args:
        input_dict (dict): Input features with keys:
            - 'Scaffold_Geometry'
            - 'Porosity_Percentage'
            - 'Immersion_Time_Days'
            - 'Mechanical_Loading'

    Returns:
        dict: Predicted values with keys:
            - 'Compressive_Stiffness_MPa'
            - 'Weight_Loss_Percentage'
            - 'Water_Absorption_Percentage'
    """
    df = pd.DataFrame([input_dict])
    X_processed = preprocessor.transform(df)
    predictions = model.predict(X_processed)[0]

    return {
        "Compressive_Stiffness_MPa": round(predictions[0], 3),
        "Weight_Loss_Percentage": round(predictions[1], 3),
        "Water_Absorption_Percentage": round(predictions[2], 3)
    }
