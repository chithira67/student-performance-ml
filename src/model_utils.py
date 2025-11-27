# model_utils.py

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

def split_data(X, y):
    """Train-test split."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_model(preprocessor, X_train, y_train):
    """
    Train a Random Forest Classifier with preprocessing pipeline.
    """
    model = RandomForestClassifier(n_estimators=200, random_state=42)

    from sklearn.pipeline import Pipeline
    clf = Pipeline(steps=[("preprocessor", preprocessor),
                         ("model", model)])
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):
    """Evaluate model."""
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    return acc, report


def save_model(clf, filename="student_performance_model.pkl"):
    """Save model to disk."""
    joblib.dump(clf, filename)


if __name__ == "__main__":
    print("Model utils ready.")
