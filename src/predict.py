import os
import joblib
import pandas as pd

# Define paths to model and preprocessor (relative to project root)
model_path = os.path.join("outputs", "models", "printability_model.pkl")
preprocessor_path = os.path.join("outputs", "models", "preprocessor.pkl")

# Load once (global)
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

def predict_printability(input_data: dict) -> int:
    """
    Predict the printability of a given bio-ink formulation.
    Applies hard rule: if TG_min or PS == 0, return Not Printable.

    Args:
        input_data (dict): Dictionary with all feature values.
    Returns:
        int: 1 if printable, 0 if not printable.
    """
    # Rule-based override
    if float(input_data.get("TG_min", 1)) == 0.0 or float(input_data.get("PS", 1)) == 0.0:
        return 0  # Not printable due to critical parameter being zero

    # Format input
    df = pd.DataFrame([input_data])
    df = df[list(preprocessor.feature_names_in_)]  # Match training feature order

    # Transform and predict
    X = preprocessor.transform(df)
    return model.predict(X)[0]
