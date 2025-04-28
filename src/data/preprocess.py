"""
Module for preprocessing the bank loan dataset.
"""
import numpy as np
import pandas as pd


def check_missing_values(data):
    """
    Check for missing values in the dataset.

    Args:
        data (pd.DataFrame): The dataset to check

    Returns:
        pd.DataFrame: DataFrame containing missing value information
    """
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100

    missing_info = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percentage
    }).sort_values('Missing Count', ascending=False)

    return missing_info[missing_info['Missing Count'] > 0]


def check_skewness(data):
    """
    Check skewness of numerical features.

    Args:
        data (pd.DataFrame): The dataset to check

    Returns:
        pd.Series: Series containing skewness values of numerical features
    """
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
    return data[numeric_features].skew().sort_values(ascending=False)


def encode_categorical_features(X):
    """
    One-hot encode categorical features.
    Here the column 'ed' is numeric, but it is actually a categorical feature.
    Note that we can drop the first categorical value with no loss of information.

    Args:
        X (pd.DataFrame): Feature matrix

    Returns:
        pd.DataFrame: Feature matrix with categorical features encoded
    """
    X_encoded = X.copy()

    # One-hot encode the education feature
    education_encoded = pd.get_dummies(X_encoded['ed'], prefix='ed', drop_first=True)
    X_encoded = pd.concat([X_encoded.drop('ed', axis=1), education_encoded], axis=1)

    return X_encoded


def scale_skewed_features(X):
    """
    Apply log transformation to skewed features.

    Args:
        X (pd.DataFrame): Feature matrix

    Returns:
        pd.DataFrame: Feature matrix with skewed features scaled
    """
    X_scaled = X.copy()

    # Scale skewed features based on prior analysis
    X_scaled['income'] = np.log1p(X_scaled['income'])
    # Additional scaling can be uncommented if needed
    # X_scaled['creddebt'] = np.log1p(X_scaled['creddebt'])
    # X_scaled['othdebt'] = np.log1p(X_scaled['othdebt'])

    return X_scaled


def preprocess_data(X, scale_features=True, encode_categorical=True):
    """
    Apply preprocessing steps to the feature matrix.
    Here I added the preprocessing steps as parameters to make it easy to compare the result of not preprocessing the dataset.

    Args:
        X (pd.DataFrame): Raw feature matrix
        scale_features (bool): Whether to scale skewed features
        encode_categorical (bool): Whether to encode categorical features

    Returns:
        pd.DataFrame: Preprocessed feature matrix
    """
    X_processed = X.copy()

    if encode_categorical:
        X_processed = encode_categorical_features(X_processed)

    if scale_features:
        X_processed = scale_skewed_features(X_processed)

    return X_processed


def analyze_class_distribution(y):
    """
    Analyze the distribution of the target variable.

    Args:
        y (pd.Series): Target variable

    Returns:
        dict: Dictionary containing class distribution information
    """
    class_counts = y.value_counts()
    default_rate = class_counts[1] / len(y)

    return {
        'class_counts': class_counts,
        'default_rate': default_rate,
        'class_distribution': class_counts / len(y) * 100
    }


if __name__ == "__main__":
    # This code will run if the script is executed directly
    from load_data import load_data, get_feature_target_split

    data = load_data()
    if data is not None:
        X, y = get_feature_target_split(data)

        # Check for missing values
        missing = check_missing_values(data)
        print("Missing values in the dataset:")
        print(missing if not missing.empty else "No missing values found.")

        # Check skewness
        skewness = check_skewness(data)
        print("\nSkewness of numerical features:")
        print(skewness)

        # Analyze class distribution
        class_info = analyze_class_distribution(y)
        print("\nClass distribution:")
        print(class_info['class_counts'])
        print(f"Default rate: {class_info['default_rate']:.2%}")

        # Preprocess data
        X_processed = preprocess_data(X)
        print(f"\nPreprocessed data shape: {X_processed.shape}")
        print("First 5 rows of preprocessed data:")
        print(X_processed.head())