"""
Module for implementing an XGBoost classifier for loan default prediction.
"""
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import time
from datetime import timedelta


def train_xgboost_with_grid_search(X_train, y_train, param_grid=None, n_splits=5, n_estimators=200, random_state=42,eval_metric='aucpr'):
    """
    Train an XGBoost classifier with hyperparameter tuning using GridSearchCV.

    XGBoost is well-suited for this loan default prediction problem because:
    1. It often outperforms other algorithms for structured/tabular data
    2. It handles missing values and is robust to outliers
    3. It has built-in regularization that helps prevent overfitting
    4. It's efficient and scalable

    For loan default prediction, tuning these specific parameters is important:
    - learning_rate: Controls how quickly the model learns
    - max_depth: Controls complexity/potential overfitting
    - min_child_weight: Helps with class imbalance issues

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        param_grid (dict, optional): Parameter grid for GridSearchCV
        cv (int): Number of cross-validation folds

    Returns:
        dict: Results including best model and best parameters
    """
    # Default parameter grid based on original analysis
    if param_grid is None:
        param_grid = {
            'learning_rate': [0.05, 0.1, 0.2, 0.3],
            'max_depth': [3, 4, 5],
            'min_child_weight': [15, 20, 30],
            'scale_pos_weight': [1, 3, 5, 10]
        }

    # Initialize the base model
    base_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        eval_metric=eval_metric
    )

    # Create a stratified k-fold object
    stratified_cv = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=42)

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=stratified_cv,
        verbose=1
    )

    # Fit the grid search to the data
    print("Starting hyperparameter tuning with GridSearchCV, this may take a while...")

    # Start timing before grid search
    start_time = time.time()
    print(f"Grid search started at: {time.strftime('%H:%M:%S')}")


    grid_search.fit(X_train, y_train)

    # End timing after grid search completes
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Grid search completed at: {time.strftime('%H:%M:%S')}")
    print(f"Total time elapsed: {timedelta(seconds=elapsed_time)}")
    print(f"Time per fit: {timedelta(seconds=elapsed_time / 6912)} (average across {6912} total fits)")

    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Extract best model
    best_model = grid_search.best_estimator_

    return {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'grid_search': grid_search
    }


def evaluate_xgboost(model, X_test, y_test, model_name="XGBoost"):
    """
    Evaluate a trained XGBoost model on the test data.

    Args:
        model (xgb.XGBClassifier): Trained XGBoost model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        model_name (str): Name to display for the model

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
    print(f"{model_name} Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Return metrics
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
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

        # Train XGBoost with GridSearchCV
        print("\n--- Training XGBoost with GridSearchCV ---")
        xgb_results = train_xgboost_with_grid_search(
            X_train,
            y_train,
            param_grid={
                'learning_rate': [0.03, 0.05, 0.1]
                , 'max_depth': [2, 3]
                , 'min_child_weight': [10, 20]
                , 'scale_pos_weight': [0.5, 1, 2, 3]
            },
            eval_metric = 'aucpr',
            n_splits=3,
            n_estimators=200,
        )

        # Evaluate model
        print("\n--- Evaluating XGBoost Model ---")
        eval_results = evaluate_xgboost(
            xgb_results['model'],
            X_test,
            y_test
        )

        print("\nXGBoost model training and evaluation complete")