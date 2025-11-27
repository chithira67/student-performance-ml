# data_cleaning.py

import pandas as pd
from sklearn.impute import SimpleImputer

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    df = pd.read_csv("C:\Users\DELL\Desktop\student-performance-ml\data\raw\student-mat.csv")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numerical with median & categorical with most frequent."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates()


def clean_dataset(path: str) -> pd.DataFrame:
    """Full cleaning pipeline."""
    df = load_data(path)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    return df


if __name__ == "__main__":
    cleaned_df = clean_dataset("student-mat.csv")
    cleaned_df.to_csv("cleaned_student_mat.csv", index=False)
