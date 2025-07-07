import pandas as pd
import joblib
from src.model import predict
from src.data_preprocessing import preprocess_data


# Load model and preprocessor
model = joblib.load('outputs/models/printability_model.pkl')
preprocessor = joblib.load('outputs/models/preprocessor.pkl')

# Input data: only features known at prediction time
input_data = pd.DataFrame([{
   'Gelatin_pct': 15,              # Within optimal range (14–16%)
    'Silk_pct': 4.5,                # Ideal silk concentration
    'LH': 0.68,                     # Balanced layer height
    'PP': 55,                       # Moderate pressure
    'PS': 10,                       # Safe, stable speed
    'T': 23.5,                      # Well within ideal temp
    'TG_min': 6,                    # Balanced printing duration
    'Used_crosslinker': 1,         
    'Needle': '22G',               # Reliable and commonly used
    'Remarks': 'unknown'
}])

# Ensure columns match preprocessor
expected_order = list(preprocessor.feature_names_in_)
input_data = input_data[expected_order]  # ✅ reorder to be safe

# Preprocess and predict
processed_input = preprocessor.transform(input_data)
prediction = model.predict(processed_input)

# Output result
print("Predicted Printability:", "Printable ✅" if prediction[0] == 1 else "Not Printable ❌")
