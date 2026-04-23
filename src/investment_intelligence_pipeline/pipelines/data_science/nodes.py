import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import logging

def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into features and targets training and test sets."""
    
    # We only need the cleaned text and the label
    X = data["cleaned_sentence"]
    y = data["sentiment_label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    
    # We return them packaged nicely for the next node
    train_data = pd.DataFrame({"cleaned_sentence": X_train, "sentiment_label": y_train})
    test_data = pd.DataFrame({"cleaned_sentence": X_test, "sentiment_label": y_test})
    
    return train_data, test_data

def train_model(train_data: pd.DataFrame, parameters: Dict):
    """Vectorizes text and trains the Random Forest model."""
    
    vectorizer = TfidfVectorizer(max_features=parameters["tfidf_max_features"])
    X_train_vec = vectorizer.fit_transform(train_data["cleaned_sentence"])
    y_train = train_data["sentiment_label"]
    
    logger = logging.getLogger(__name__)
    logger.info("Training Random Forest Sentiment Model...")
    
    # Use the parameters here!
    classifier = RandomForestClassifier(
        n_estimators=parameters["n_estimators"],
        max_depth=parameters["max_depth"],
        random_state=parameters["random_state"]
    )
    classifier.fit(X_train_vec, y_train)
    
    pipeline_model = {"vectorizer": vectorizer, "classifier": classifier}
    return pipeline_model

def evaluate_model(pipeline_model: dict, test_data: pd.DataFrame) -> Dict[str, float]:
    """Evaluates the model on the test set."""
    
    vectorizer = pipeline_model["vectorizer"]
    classifier = pipeline_model["classifier"]
    
    X_test_vec = vectorizer.transform(test_data["cleaned_sentence"])
    y_test = test_data["sentiment_label"]
    
    # Predict
    y_pred = classifier.predict(X_test_vec)
    
    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Model Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    
    return {"accuracy": accuracy}