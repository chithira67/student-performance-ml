# feature_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary target variable: pass/fail based on final grade G3.
    """
    df["pass"] = (df["G3"] >= 10).astype(int)
    return df


def build_preprocessing_pipeline(df: pd.DataFrame):
    """Build preprocessing pipeline (scaling + one-hot encoding)."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop("pass")
    cat_cols = df.select_dtypes(include=['object']).columns

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    return preprocessor, num_cols, cat_cols


def preprocess_features(df: pd.DataFrame):
    """Create target + preprocessing pipeline."""
    df = create_target(df)
    preprocessor, num_cols, cat_cols = build_preprocessing_pipeline(df)
    X = df.drop(["pass"], axis=1)
    y = df["pass"]
    return X, y, preprocessor


if __name__ == "__main__":
    df = pd.read_csv("cleaned_student_mat.csv")
    X, y, preprocessor = preprocess_features(df)
    print("Feature engineering complete.")
