import os
import joblib
import pandas as pd

# --- Paths ---
MODEL_DIR = os.path.join("degradation_project", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "degradation_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

# --- Load model and preprocessor ---
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# --- Example Input ---
input_data = {
   "Scaffold_Geometry": "Body Centered",
    "Porosity_Percentage": 50.0,           # Low porosity → stronger
    "Immersion_Time_Days": 15,             # Short immersion
    "Mechanical_Loading": 0                # No extra stress
}

# --- Convert to DataFrame ---
df = pd.DataFrame([input_data])

# --- Preprocess ---
X_processed = preprocessor.transform(df)

# --- Predict ---
predictions = model.predict(X_processed)[0]

# --- Display ---
print("\n📊 Predicted Scaffold Degradation Outcomes:\n")
print(f"🔧 Compressive Stiffness (MPa):     {predictions[0]:.2f}")
print(f"🧪 Weight Loss (%):                 {predictions[1]:.2f}")
print(f"💧 Water Absorption (%):           {predictions[2]:.2f}")
print("\n✅ Prediction complete.\n")
