"""
Module for implementing a Random Forest classifier for loan default prediction.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_random_forest(X_train, y_train, n_estimators=200, random_state=42):
    """
    Train a Random Forest classifier for loan default prediction.

    Random Forest is a good choice for this problem because:
    1. It handles non-linear relationships between features
    2. It's robust to outliers and non-normally distributed data
    3. It provides feature importance measures
    4. It generally performs well even with default parameters

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        n_estimators (int): Number of trees in the forest
        random_state (int): Random seed for reproducibility

    Returns:
        RandomForestClassifier: Trained Random Forest model
    """
    # Initialize the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    # Train the model
    model.fit(X_train, y_train)

    print(f"Random Forest model trained with {n_estimators} trees")

    return model


def evaluate_random_forest(model, X_test, y_test):
    """
    Evaluate a trained Random Forest model on the test data.

    Args:
        model (RandomForestClassifier): Trained Random Forest model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target

    Returns:
        dict: Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Display results
    print(f"Random Forest Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Get feature importance
    feature_importance = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=False)


    # Return metrics
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'feature_importance': feature_importance,
        'predictions': y_pred
    }


if __name__ == "__main__":
    # Add parent directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(parent_dir)

    # Import from other modules
    from data.load_data import load_data, get_feature_target_split
    from data.preprocess import preprocess_data
    from features.build_features import create_train_test_split

    # Load and preprocess data
    data = load_data()
    if data is not None:
        X, y = get_feature_target_split(data)
        X_processed = preprocess_data(X)

        # Split data
        X_train, X_test, y_train, y_test = create_train_test_split(X_processed, y)

        # Train and evaluate Random Forest model
        rf_model = train_random_forest(X_train, y_train)
        results = evaluate_random_forest(rf_model, X_test, y_test)

        print("\nRandom Forest model training and evaluation complete")