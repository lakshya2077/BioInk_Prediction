import os
import pandas as pd
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.utils import setup_logging

# --- Set Up Logging ---
setup_logging()
logging.info("🛠️ Starting training pipeline...")
print("🛠️ Starting training pipeline...")

# --- Paths ---
DATA_PATH = os.path.join("degradation_project", "data", "degradation_dataset.csv")
MODEL_DIR = os.path.join("degradation_project", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "degradation_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

# --- Load Data ---
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# --- Features and Targets ---
X = df[["Scaffold_Geometry", "Porosity_Percentage", "Immersion_Time_Days", "Mechanical_Loading"]]
y = df[[
    "Compressive_Stiffness_MPa",
    "Weight_Loss_Percentage",
    "Water_Absorption_Percentage"
]]

# --- Preprocessing ---
print("🧹 Setting up preprocessing pipeline...")
categorical_cols = ["Scaffold_Geometry"]
numeric_cols = ["Porosity_Percentage", "Immersion_Time_Days", "Mechanical_Loading"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

# --- Model Pipeline ---
print("⚙️  Building model pipeline...")
base_model = RandomForestRegressor(n_estimators=100, random_state=42)
multioutput_model = MultiOutputRegressor(base_model)

pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", multioutput_model)
])

# --- Split ---
print("🔀 Splitting dataset into train and test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train ---
print("🏋️ Training the model...")
pipeline.fit(X_train, y_train)
logging.info("✅ Model training complete.")
print("✅ Model training complete.")

# --- Evaluate ---
print("\n📊 Evaluation Metrics:")
y_pred = pipeline.predict(X_test)
for i, col in enumerate(y.columns):
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"• {col}: MSE = {mse:.3f}, R² = {r2:.3f}")
    logging.info(f"{col} → MSE: {mse:.3f} | R²: {r2:.3f}")

# --- Save ---
print("\n💾 Saving model and preprocessor...")
joblib.dump(pipeline.named_steps["regressor"], MODEL_PATH)
joblib.dump(preprocessor, PREPROCESSOR_PATH)
logging.info(f"✅ Model saved to: {MODEL_PATH}")
logging.info(f"✅ Preprocessor saved to: {PREPROCESSOR_PATH}")
print("✅ All done! Artifacts saved in:", MODEL_DIR)
