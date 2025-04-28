"""
Module for loading the bank loan dataset.
"""

import os
import pandas as pd


def load_data(filepath=None):
    """
    Load the bank loan dataset from the specified filepath.

    Args:
        filepath (str, optional): Path to the CSV file. If None, uses the default path.

    Returns:
        pd.DataFrame: The loaded dataset
    """
    if filepath is None:
        # Construct path relative to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        filepath = os.path.join(project_root, 'data', 'raw', 'bank_loan.csv')

    # Load the dataset
    try:
        data = pd.read_csv(filepath)
        print(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def get_feature_target_split(data):
    """
    Split the dataset into features (X) and target variable (y).

    Args:
        data (pd.DataFrame): The loaded dataset

    Returns:
        tuple: (X, y) containing features and target variable
    """
    if data is None:
        return None, None

    X = data.drop('default', axis=1)
    y = data['default']

    return X, y


def display_data_info(data):
    """
    Display basic information about the dataset.

    Args:
        data (pd.DataFrame): The dataset to display information about
    """
    if data is None:
        print("No data to display.")
        return

    print(f"Features in the dataset:\n {data.columns}")
    print(f"\nFirst 5 rows of data: \n {data.head()}")
    print(f"\nStatistical properties of the data:\n{data.describe()}")
    print(f"\nData type for each column:\n {data.dtypes}")


if __name__ == "__main__":
    # This code will run if the script is executed directly
    data = load_data()
    if data is not None:
        display_data_info(data)
        X, y = get_feature_target_split(data)
        print(f"\nShape of feature matrix X: {X.shape}")
        print(f"Shape of target variable y: {y.shape}")