import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(path):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(path)

def preprocess_data(df, target_column, fit=True, preprocessor=None):
    """
    Preprocess the dataset: handle missing values, encode, scale, and split.

    Parameters:
        df (pd.DataFrame): The input data.
        target_column (str): The name of the target column.
        fit (bool): If True, fit the transformer; otherwise, use an existing one.
        preprocessor: Pre-fitted transformer to use if fit=False.

    Returns:
        - If fit=True: X_train, X_test, y_train, y_test, preprocessor
        - If fit=False: X_processed
    """
    
    # Only drop rows with missing target during training
    if fit:
        df = df.dropna(subset=[target_column])
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df.copy()
        y = None

    # Remove 'Remarks' only during prediction
    if not fit and 'Remarks' in X.columns:
        X = X.drop(columns=['Remarks'])

    # Detect column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    if fit:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        X_processed = preprocessor.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test, preprocessor

    else:
        if preprocessor is None:
            raise ValueError("Preprocessor must be provided when fit=False")
        X_processed = preprocessor.transform(X)
        return X_processed
