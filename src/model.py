from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

def train_model(X_train, y_train, model_type="random_forest"):
    """
    Train a machine learning model.

    Parameters:
        X_train (array-like): Training features
        y_train (array-like): Training labels
        model_type (str): Type of model to train ("random_forest" supported)

    Returns:
        Trained model object
    """
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model


def predict(new_data, model, preprocessor):
    """
    Predict the output label for new raw input data.

    Parameters:
        new_data (pd.DataFrame or dict): Raw input data
        model: Trained ML model
        preprocessor: Fitted preprocessing pipeline

    Returns:
        np.ndarray: Predicted labels
    """
    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])
    elif not isinstance(new_data, pd.DataFrame):
        raise TypeError("new_data must be a dict or pandas DataFrame")

    # ❗ Remove 'Remarks' if present (not used during prediction)
    if 'Remarks' in new_data.columns:
        new_data = new_data.drop(columns=['Remarks'])

    transformed = preprocessor.transform(new_data)
    return model.predict(transformed)


def save_model(model, path):
    """
    Save a trained model to a file.

    Parameters:
        model: Trained model
        path (str): File path to save the model
    """
    joblib.dump(model, path)


def load_model(path):
    """
    Load a trained model from a file.

    Parameters:
        path (str): File path to load the model from

    Returns:
        Loaded model
    """
    return joblib.load(path)
