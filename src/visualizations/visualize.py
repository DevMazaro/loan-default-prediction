"""
Module for creating visualizations for the bank loan dataset.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def save_figure(fig, filename, directory=None):
    """
    Save a matplotlib figure to a file.

    Args:
        fig (matplotlib.figure.Figure): Figure to save
        filename (str): Name of the file
        directory (str, optional): Directory to save the file in
    """
    if directory is None:
        # Default to project's visualization directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        directory = os.path.join(project_root, 'visualizations')

    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Save figure
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Figure saved to {filepath}")
    plt.close(fig)


def plot_correlation_heatmap(data, figsize=(12, 10), save=True):
    """
    Create a correlation heatmap for numerical variables.

    This visualization helps identify relationships between features like:
    - Debt-to-income ratio correlation with default status
    - Relationship between income and debt levels
    - Patterns in employment years and credit behavior

    Args:
        data (pd.DataFrame): Dataset to visualize
        figsize (tuple): Figure size (width, height)
        save (bool): Whether to save the figure to a file

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    correlation_matrix = data.corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    plt.title("Correlation Heatmap of Loan Features", fontsize=16)

    if save:
        save_figure(fig, "correlation_heatmap.png")

    return fig


def plot_pairplot(data, variables=None, hue='default', save=True):
    """
    Create a pairplot to visualize relationships between variables.

    This shows how defaulters differ from non-defaulters across multiple dimensions:
    - Age distributions for defaulters vs non-defaulters
    - Income patterns across default status
    - Employment years and default patterns
    - Debt-to-income ratio differences

    Args:
        data (pd.DataFrame): Dataset to visualize
        variables (list): List of variables to include in the pairplot
        hue (str): Column to use for coloring points
        save (bool): Whether to save the figure to a file

    Returns:
        seaborn.axisgrid.PairGrid: The created pairplot
    """
    if variables is None:
        variables = ['age', 'employ', 'income', 'debtinc']

    pairplot = sns.pairplot(data, vars=variables, hue=hue)
    plt.suptitle("Relationships Between Key Loan Features", y=1.02, fontsize=16)

    if save:
        save_figure(pairplot.fig, "pairplot.png")

    return pairplot


def plot_boxplot(data, column, horizontal=True, save=True):
    """
    Create a boxplot for a specific column.

    Boxplots help identify:
    - The distribution and range of values
    - Presence of outliers that might affect model performance
    - Median and quartile information

    Args:
        data (pd.DataFrame): Dataset to visualize
        column (str): Column to create boxplot for
        horizontal (bool): Whether to plot horizontally
        save (bool): Whether to save the figure to a file

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if horizontal:
        sns.boxplot(x=column, data=data, ax=ax)
        plt.xlabel(column, fontsize=12)
    else:
        sns.boxplot(y=column, data=data, ax=ax)
        plt.ylabel(column, fontsize=12)

    plt.title(f"Distribution of {column}", fontsize=16)

    if save:
        save_figure(fig, f"boxplot_{column}.png")

    return fig


def plot_class_distribution(y, save=True):
    """
    Create a bar plot of the class distribution.

    This visualization shows the imbalance between defaulters and non-defaulters,
    which is important to consider for model evaluation metrics.

    Args:
        y (pd.Series): Target variable
        save (bool): Whether to save the figure to a file

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    class_counts = y.value_counts().reset_index()
    class_counts.columns = ['Default', 'Count']

    # Add percentage labels
    class_counts['Percentage'] = class_counts['Count'] / class_counts['Count'].sum() * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = sns.barplot(x='Default', y='Count', hue='Default', data=class_counts, palette='viridis', ax=ax, legend=False)

    # Add percentage labels above bars
    for i, (_, row) in enumerate(class_counts.iterrows()):
        ax.text(i, row['Count'] + 5, f"{row['Percentage']:.1f}%",
                ha='center', va='bottom', fontsize=12)

    ax.set_xlabel('Loan Default Status (1=Default)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Class Distribution of Loan Default Status', fontsize=16)

    # Set x-tick labels
    ticks = range(len(class_counts))
    ax.set_xticks(ticks)
    ax.set_xticklabels(['No Default (0)', 'Default (1)'])

    if save:
        save_figure(fig, "class_distribution.png")

    return fig


def plot_feature_importance(feature_importance, top_n=10, save=True):
    """
    Create a bar plot of feature importance.

    Shows which features the model considers most important for predicting loan defaults,
    helping to validate domain expertise and guide feature engineering.

    Args:
        feature_importance (pd.Series): Feature importance scores
        top_n (int): Number of top features to display
        save (bool): Whether to save the figure to a file

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    top_features = feature_importance.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.barplot(x=top_features.values, y=top_features.index, palette='viridis', ax=ax)

    ax.set_title(f'Top {top_n} Important Features for Loan Default Prediction', fontsize=16)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)

    if save:
        save_figure(fig, "feature_importance.png")

    return fig


def plot_confusion_matrix(cm, class_names, save=True, normalize=False, filename="confusion_matrix.png"):
    """
    Create a visualization of a confusion matrix.

    Confusion matrices help understand:
    - True positives (correctly predicted defaults)
    - False positives (non-defaults predicted as defaults)
    - True negatives (correctly predicted non-defaults)
    - False negatives (defaults predicted as non-defaults)

    For loan default prediction, false negatives are particularly costly.

    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list): List of class names
        save (bool): Whether to save the figure to a file
        normalize (bool): Whether to normalize the confusion matrix
        filename (str): Name of the file to save

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=16)

    if save:
        save_figure(fig, filename)

    return fig


def plot_feature_distributions(data, feature, by='default', save=True):
    """
    Plot the distribution of a feature by default status.

    This helps visualize how feature distributions differ between defaulters
    and non-defaulters, revealing patterns that might inform feature engineering.

    Args:
        data (pd.DataFrame): Dataset to visualize
        feature (str): Feature to visualize
        by (str): Column to group by (usually default status)
        save (bool): Whether to save the figure to a file

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Map default status to descriptive labels
    data_copy = data.copy()
    if by == 'default':
        data_copy['default_status'] = data_copy['default'].map({0: 'No Default', 1: 'Default'})
        by = 'default_status'

    # Plot distributions
    sns.histplot(data=data_copy, x=feature, hue=by, kde=True, ax=ax)

    plt.title(f'Distribution of {feature} by Default Status', fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Count', fontsize=12)

    if save:
        save_figure(fig, f"distribution_{feature}.png")

    return fig


if __name__ == "__main__":
    # This code will run if the script is executed directly
    import sys
    import os

    # Add parent directory to path to import modules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(parent_dir)

    from data.load_data import load_data, get_feature_target_split

    # Load data
    data = load_data()

    if data is not None:
        X, y = get_feature_target_split(data)

        # Generate example visualizations
        plot_correlation_heatmap(data)
        plot_pairplot(data)
        plot_boxplot(data, 'income')
        plot_class_distribution(y)
        plot_feature_distributions(data, 'debtinc')