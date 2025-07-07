import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def load_degradation_data(csv_path: str) -> pd.DataFrame:
    """
    Load the degradation dataset from CSV.
    """
    return pd.read_csv(csv_path)


def get_feature_columns():
    """
    Returns a list of input feature columns for the model.
    """
    return ['Scaffold_Geometry', 'Porosity_Percentage', 'Immersion_Time_Days', 'Mechanical_Loading']


def get_target_columns():
    """
    Returns a list of output target columns for degradation prediction.
    """
    return ['Compressive_Stiffness_MPa', 'Weight_Loss_Percentage', 'Water_Absorption_Percentage']


def build_preprocessor():
    """
    Returns a fitted ColumnTransformer that handles categorical encoding.
    Used both during training and during prediction preprocessing.
    """
    categorical_features = ['Scaffold_Geometry']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'  # Keep numeric columns as-is
    )
    return preprocessor
