import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Loads the degradation dataset."""
    return pd.read_csv(csv_path)


def build_pipeline() -> Pipeline:
    """
    Constructs a preprocessing + model pipeline.
    Returns a sklearn Pipeline with a ColumnTransformer and MultiOutputRegressor.
    """
    categorical_features = ['Scaffold_Geometry']
    numeric_features = ['Porosity_Percentage', 'Immersion_Time_Days', 'Mechanical_Loading']

    # Preprocessor
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ], remainder='passthrough')  # Keep numeric columns as-is

    # Base model
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_model)

    # Full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline


def train_and_save_model(data_path: str, model_out_path: str, preprocessor_out_path: str) -> None:
    """
    Trains the pipeline and saves the model and preprocessor separately.
    """
    df = load_dataset(data_path)

    # Features and targets
    X = df[['Scaffold_Geometry', 'Porosity_Percentage', 'Immersion_Time_Days', 'Mechanical_Loading']]
    y = df[['Compressive_Stiffness_MPa', 'Weight_Loss_Percentage', 'Water_Absorption_Percentage']]

    # Split (optional — you can skip if training on full data)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Save entire pipeline or split if needed
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    joblib.dump(pipeline.named_steps['model'], model_out_path)
    joblib.dump(pipeline.named_steps['preprocessor'], preprocessor_out_path)
    print("✅ Model and preprocessor saved.")


def load_model_and_preprocessor(model_path: str, preprocessor_path: str):
    """
    Loads the saved model and preprocessor.
    Returns:
        model: Trained MultiOutputRegressor
        preprocessor: Trained ColumnTransformer
    """
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor
