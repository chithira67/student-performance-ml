# data_cleaning.py

import pandas as pd
from sklearn.impute import SimpleImputer

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    df = pd.read_csv(path, sep=';')
    # Convert numeric columns that might be read as strings due to quotes
    numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
                    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 
                    'absences', 'G1', 'G2', 'G3']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numerical with median & categorical with most frequent."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols] = pd.DataFrame(
            num_imputer.fit_transform(df[num_cols]),
            columns=num_cols,
            index=df.index
        )
    
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = pd.DataFrame(
            cat_imputer.fit_transform(df[cat_cols]),
            columns=cat_cols,
            index=df.index
        )

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
    import os
    data_path = os.path.join("data", "raw", "student-mat.csv")
    cleaned_df = clean_dataset(data_path)
    output_path = os.path.join("data", "processed", "cleaned_student_mat.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
