# feature_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary target variable: pass/fail based on final grade G3.
    """
    df = df.copy()  # Avoid modifying original dataframe
    df["pass"] = (df["G3"] >= 10).astype(int)
    return df


def build_preprocessing_pipeline(df: pd.DataFrame):
    """Build preprocessing pipeline (scaling + one-hot encoding)."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if "pass" in num_cols:
        num_cols.remove("pass")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    transformers = []
    
    if len(num_cols) > 0:
        numeric_transformer = StandardScaler()
        transformers.append(("num", numeric_transformer, num_cols))
    
    if len(cat_cols) > 0:
        # Use sparse=False for older sklearn versions, sparse_output=False for newer
        try:
            categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
        transformers.append(("cat", categorical_transformer, cat_cols))

    if len(transformers) == 0:
        raise ValueError("No numeric or categorical columns found for preprocessing")
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    return preprocessor, num_cols, cat_cols


def preprocess_features(df: pd.DataFrame):
    """Create target + preprocessing pipeline."""
    df = create_target(df)
    X = df.drop(["pass"], axis=1)
    y = df["pass"]
    # Build pipeline on X (without target column)
    preprocessor, num_cols, cat_cols = build_preprocessing_pipeline(X)
    return X, y, preprocessor


if __name__ == "__main__":
    import os
    data_path = os.path.join("data", "processed", "cleaned_student_mat.csv")
    df = pd.read_csv(data_path)
    X, y, preprocessor = preprocess_features(df)
    print("Feature engineering complete.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
