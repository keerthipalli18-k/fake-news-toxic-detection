import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from utils.preprocessing import clean_text


# Project paths
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "datasets" / "WELFake_Dataset.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"


# Train Fake News Detection Model
def train_model():

    print("Loading dataset...")

    data = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully!")

    print("Dataset shape:", data.shape)

    # Check columns
    required_columns = ["title", "text", "label"]

    for column in required_columns:

        if column not in data.columns:

            raise ValueError(f"Missing column: {column}")

    # Remove missing values
    data = data.dropna(
        subset=["title", "text", "label"]
    ).copy()

    # Combine title and text
    data["content"] = (
        data["title"].astype(str)
        + " "
        + data["text"].astype(str)
    )

    # Clean text
    print("Cleaning news articles...")

    data["content"] = data["content"].apply(clean_text)

    data = data[data["content"].str.len() > 0]

    # Features and labels
    X = data["content"]

    y = data["label"].astype(int)

    # Check labels
    print("Available labels:", sorted(y.unique()))

    if set(y.unique()) != {0, 1}:
        raise ValueError("Dataset must contain labels 0 and 1.")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Create machine learning pipeline
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
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )

    ])

    # Train
    print("\nTraining Fake News Model...")

    model.fit(X_train, y_train)

    # Test
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nTraining completed!")

    print("Model Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Fake News", "Real News"]
        )
    )

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_PATH, compress=3)

    print("\nModel saved successfully!")

    print("Saved location:", MODEL_PATH)


# Predict news
def predict_news(news_text):

    if not news_text or not news_text.strip():

        raise ValueError("Please enter news text.")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Model not found. Train the model first."
        )

    model = joblib.load(MODEL_PATH)

    cleaned_text = clean_text(news_text)

    prediction = model.predict([cleaned_text])[0]

    probabilities = model.predict_proba([cleaned_text])[0]

    confidence = max(probabilities) * 100

    if prediction == 0:

        result = "Fake News"

    else:

        result = "Real News"

    return result, round(confidence, 2)


# Main execution
if __name__ == "__main__":

    train_model()