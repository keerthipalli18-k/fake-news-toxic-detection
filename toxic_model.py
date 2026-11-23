
import pandas as pd
import joblib
import re

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# Project paths
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "datasets" / "toxic" / "train.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "toxic_model.pkl"


# Toxicity categories
LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


# Detection threshold
THRESHOLD = 0.50


# Text preprocessing
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Train Toxic Content Model
def train_model():

    print("Loading Toxic Comments Dataset...")

    data = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully!")
    print("Dataset shape:", data.shape)

    # Check required columns
    required_columns = ["comment_text"] + LABELS

    for column in required_columns:

        if column not in data.columns:
            raise ValueError(f"Missing column: {column}")

    # Remove missing comments
    data = data.dropna(subset=["comment_text"]).copy()

    # Clean comments
    print("Cleaning comments...")

    data["comment_text"] = data["comment_text"].apply(clean_text)

    # Remove empty comments
    data = data[data["comment_text"].str.len() > 0]

    # Features and labels
    X = data["comment_text"]

    y = data[LABELS].astype(int)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Create model
    model = Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                max_features=30000,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True
            )
        ),

        (
            "classifier",
            OneVsRestClassifier(
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear"
                )
            )
        )

    ])

    # Train model
    print("\nTraining Toxic Content Model...")

    model.fit(X_train, y_train)

    print("\nTraining completed!")

    # Predictions
    predictions = model.predict(X_test)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=LABELS,
            zero_division=0
        )
    )

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH, compress=3)

    print("\nToxic Model saved successfully!")
    print("Saved location:", MODEL_PATH)


# Predict toxic content
def predict_toxicity(comment):

    if not comment or not comment.strip():

        raise ValueError("Please enter a comment.")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Toxic model not found. Train the model first."
        )

    # Load model
    model = joblib.load(MODEL_PATH)

    # Clean input
    cleaned_comment = clean_text(comment)

    if not cleaned_comment:
        raise ValueError("Please enter valid text.")

    # Get probabilities
    probabilities = model.predict_proba([cleaned_comment])[0]

    results = {}

    # Apply consistent threshold
    for i, label in enumerate(LABELS):

        confidence = float(probabilities[i]) * 100

        prediction = int(confidence >= THRESHOLD * 100)

        results[label] = {
            "prediction": prediction,
            "confidence": round(confidence, 2)
        }

    return results


# Main execution
if __name__ == "__main__":

    train_model()
