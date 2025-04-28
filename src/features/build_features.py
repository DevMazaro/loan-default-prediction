"""
Module for feature engineering on the bank loan dataset.
"""
from sklearn.model_selection import train_test_split


def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # This code will run if the script is executed directly
    import sys
    import os

    # Add parent directory to path to import modules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(parent_dir)

    from data.load_data import load_data, get_feature_target_split
    from data.preprocess import preprocess_data

    # Load and preprocess data
    data = load_data()
    if data is not None:
        X, y = get_feature_target_split(data)
        X_processed = preprocess_data(X)

        # Split data
        X_train, X_test, y_train, y_test = create_train_test_split(X_processed, y)

        # Print feature names
        print("\nFeatures after preprocessing:")
        for i, feature in enumerate(X_processed.columns):
            print(f"{i + 1}. {feature}")