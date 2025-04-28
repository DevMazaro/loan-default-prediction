"""
Loan Default Prediction

This script runs the complete loan default prediction pipeline:
1. Load and explore the bank loan dataset
2. Preprocess the data (handle missing values, encode categorical variables, scale features)
3. Train Random Forest and XGBoost models
4. Tune XGBoost hyperparameters using GridSearchCV
5. Evaluate and compare model performance
6. Generate visualizations

Author: Eduardo Mazaro
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from src.data.load_data import load_data, get_feature_target_split, display_data_info
from src.data.preprocess import check_missing_values, check_skewness, preprocess_data,analyze_class_distribution
from src.features.build_features import create_train_test_split
from src.models.random_forest_model import train_random_forest, evaluate_random_forest
from src.models.xgboost_model import train_xgboost_with_grid_search, evaluate_xgboost
from src.visualizations.visualize import plot_correlation_heatmap, plot_pairplot, plot_boxplot,plot_class_distribution, plot_confusion_matrix


def run_data_exploration(data):
    """Run data exploration steps on the dataset."""
    print("\n" + "="*50)
    print("STEP 1: DATA EXPLORATION")
    print("="*50)

    # Display basic information
    display_data_info(data)

    # Check for missing values
    missing = check_missing_values(data)
    print("\nMissing values in the dataset:")
    print(missing if not missing.empty else "No missing values found.")

    # Check skewness
    skewness = check_skewness(data)
    print("\nSkewness of numerical features:")
    print(skewness)

    # Create visualizations
    plot_correlation_heatmap(data)
    plot_pairplot(data, variables=['age', 'employ', 'income', 'debtinc'])
    plot_boxplot(data, 'income')

    # Get features and target
    X, y = get_feature_target_split(data)

    # Analyze class distribution
    class_info = analyze_class_distribution(y)
    print("\nClass distribution:")
    print(class_info['class_counts'])
    print(f"Default rate: {class_info['default_rate']:.2%}")

    plot_class_distribution(y)

    return X, y


def run_preprocessing(X, y):
    """Preprocess the data and split into train/test sets."""
    print("\n" + "="*50)
    print("STEP 2: DATA PREPROCESSING")
    print("="*50)

    # Preprocess data (encode categorical variables, scale features)
    X_processed = preprocess_data(X)
    print(f"Preprocessed data shape: {X_processed.shape}")
    print("First 5 rows of preprocessed data:")
    print(X_processed.head())

    # Split into train/test sets
    X_train, X_test, y_train, y_test = create_train_test_split(X_processed, y)

    return X_processed, X_train, X_test, y_train, y_test


def run_random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate a Random Forest model."""
    print("\n" + "="*50)
    print("STEP 3: RANDOM FOREST MODEL")
    print("="*50)

    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train, n_estimators=200)

    # Evaluate Random Forest
    rf_results = evaluate_random_forest(rf_model, X_test, y_test)

    # Visualize confusion matrix
    plot_confusion_matrix(
        rf_results['confusion_matrix'],
        class_names=['No Default', 'Default'],
        save=True,
        filename="confusion_matrix_rf.png"
    )

    return rf_model, rf_results


def run_xgboost(X_train, X_test, y_train, y_test):
    """Train, tune, and evaluate XGBoost model using GridSearchCV."""
    print("\n" + "="*50)
    print("STEP 4: XGBOOST MODEL")
    print("="*50)

    # Train XGBoost with GridSearchCV
    print("\n--- Training XGBoost with GridSearchCV ---")
    xgb_results = train_xgboost_with_grid_search(
        X_train, y_train,
        param_grid={
            'learning_rate': [0.05, 0.1, 0.2, 0.3],
            'max_depth': [3, 4, 5],
            'min_child_weight': [15, 20, 30]
        }
    )

    # Evaluate GridSearchCV tuned model
    print("\n--- Evaluating XGBoost Model ---")
    xgb_model = xgb_results['model']
    eval_results = evaluate_xgboost(xgb_model, X_test, y_test)

    # Plot confusion matrix for XGBoost model
    plot_confusion_matrix(
        eval_results['confusion_matrix'],
        class_names=['No Default', 'Default'],
        save=True,
        filename="confusion_matrix_xgb.png"
    )

    return xgb_model, eval_results, xgb_results['best_params']


def compare_models(rf_results, xgb_results):
    """Compare the performance of Random Forest and XGBoost models."""
    print("\n" + "="*50)
    print("STEP 5: MODEL COMPARISON")
    print("="*50)

    # Create comparison dataframe
    comparison_data = [
        {
            'Model': 'Random Forest',
            'Accuracy': rf_results['accuracy'],
            'Precision (Default)': rf_results['classification_report']['1']['precision'],
            'Recall (Default)': rf_results['classification_report']['1']['recall'],
            'F1 (Default)': rf_results['classification_report']['1']['f1-score']
        },
        {
            'Model': 'XGBoost (GridSearchCV)',
            'Accuracy': xgb_results['accuracy'],
            'Precision (Default)': xgb_results['classification_report']['1']['precision'],
            'Recall (Default)': xgb_results['classification_report']['1']['recall'],
            'F1 (Default)': xgb_results['classification_report']['1']['f1-score']
        }
    ]

    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))

    # Create comparison chart
    plt.figure(figsize=(14, 8))

    # Plot accuracy
    ax1 = plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='Accuracy', hue='Model', data=comparison_df, palette='viridis', ax=ax1, legend=False)
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim(0.7, 1.0)  # Adjust as needed
    plt.xticks(rotation=45, ha='right')

    # Plot F1, Precision, Recall for default class
    ax2 = plt.subplot(1, 2, 2)
    metrics_df = comparison_df.melt(
        id_vars=['Model'],
        value_vars=['Precision (Default)', 'Recall (Default)', 'F1 (Default)'],
        var_name='Metric', value_name='Value'
    )
    sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_df, palette='Set2', ax=ax2)
    ax2.set_title('Default Class Metrics Comparison')
    ax2.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Save the comparison chart
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save comparison to CSV
    os.makedirs('results', exist_ok=True)
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print("\nComparison saved to 'results/model_comparison.csv'")
    print("Comparison chart saved to 'visualizations/model_comparison.png'")

    return comparison_df


def main():
    """Run the complete loan default prediction pipeline."""
    print("LOAN DEFAULT PREDICTION PIPELINE")
    print("="*50)

    # Step 1: Load data
    data = load_data()
    if data is None:
        print("Error: Could not load data. Exiting.")
        return

    # Step 2: Data exploration
    X, y = run_data_exploration(data)

    # Step 3: Data preprocessing
    X_processed, X_train, X_test, y_train, y_test = run_preprocessing(X, y)

    # Step 4: Random Forest model
    rf_model, rf_results = run_random_forest(X_train, X_test, y_train, y_test)

    # Step 5: XGBoost model with GridSearchCV
    xgb_model, xgb_results, xgb_best_params = run_xgboost(X_train, X_test, y_train, y_test)

    # Step 6: Compare models
    comparison = compare_models(rf_results, xgb_results)

    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    print("\nResults are saved in the 'visualizations' and 'results' directories.")
    print(f"\nBest XGBoost parameters: {xgb_best_params}")


if __name__ == "__main__":
    main()