# Loan Default Prediction

**Author**  
Eduardo Mazaro - [LinkedIn](https://www.linkedin.com/in/eduardomazaro/)

## Project Overview
A machine learning application that predicts bank loan defaults using Random Forest and XGBoost classifiers. This project demonstrates a complete ML pipeline from data preparation to model evaluation, with a focus on handling class imbalance in financial data.

## Executive Summary
This analysis seeks to predict which bank loan customers are likely to default, allowing financial institutions to make more informed lending decisions and better manage risk. Two machine learning models were implemented and tuned:

- **Random Forest**: Achieved 83.6% accuracy with 62.2% recall for detecting defaults
- **XGBoost (tuned)**: Achieved 79.3% accuracy with 51.4% recall for detecting defaults

The models reveal that debt-to-income ratio, total credit debt, and income are the strongest predictors of loan default. Recall (the ability to identify actual defaults) is particularly critical in this context, as missing a potential default is significantly more costly to financial institutions than incorrectly flagging a good loan. With a dataset containing 26.4% defaults, our Random Forest model can correctly identify 62.2% of them, representing a substantial improvement over random guessing or basic rule-based approaches.

## Features
- Data preprocessing pipeline with categorical encoding and log transformation for skewed features
- Feature engineering focused on financial indicators
- Implementation of Random Forest and XGBoost classifiers
- Hyperparameter tuning using GridSearchCV with stratified cross-validation
- Comprehensive model evaluation with focus on imbalanced classification metrics

## Key Results

### Model Performance Comparison
| Model | Accuracy | Precision (Default) | Recall (Default) | F1 (Default) |
|-------|----------|---------------------|------------------|--------------|
| Random Forest | 0.836 | 0.767 | 0.622 | 0.687 |
| XGBoost (GridSearchCV) | 0.793 | 0.710 | 0.514 | 0.600 |

### Insights
- Random Forest consistently outperformed XGBoost across all metrics for our loan default prediction task
- Feature preprocessing significantly impacts model performance - encoding categorical variables improved the Random Forest model while having minimal impact on XGBoost
- Scaling features had negligible impact on model performance for both algorithms
- Complex parameter tuning with additional XGBoost parameters (subsample, colsample_bytree) did not improve performance, suggesting simpler models generalize better for this specific dataset

## Model Exploration and Tuning

### Feature Processing Experiments
Extensive testing was conducted to understand the impact of different preprocessing approaches:

1. **Impact of Encoding and Scaling**:
   - Random Forest performed best with categorical encoding only (86.5% accuracy, 62.2% recall)
   - XGBoost performed better with raw features (80.7% accuracy, 37.8% recall)
   - Scaling numerical features provided no notable benefit to either model

2. **Class Imbalance Handling**:
   - The dataset contained 26.4% defaults (moderately imbalanced)
   - Techniques tested: scale_pos_weight parameter in XGBoost, class_weight in Random Forest
   - Random Forest with balanced class weighting proved most effective for maximizing recall

### XGBoost Parameter Tuning
Extensive grid search was performed to optimize XGBoost parameters:

1. **Basic Parameters**:
   - Best learning_rate: 0.03
   - Best max_depth: 2
   - Best min_child_weight: 10
   - Best scale_pos_weight: 1

2. **Advanced Parameters**:
   - Adding subsample and colsample_bytree parameters reduced performance
   - Fine-tuning around optimal basic parameters did not yield significant improvements
   - Best cross-validation score was achieved with learning_rate=0.04, max_depth=2, min_child_weight=12, scale_pos_weight=0.8, but this didn't translate to better test performance

3. **Key Finding**: Simpler models with carefully tuned basic parameters outperformed more complex ones, demonstrating better generalization to unseen data.

## Future Work
- Develop a cost-benefit analysis framework to quantify the financial impact of false negatives vs. false positives
- Incorporate additional customer behavioral features that may signal financial distress
- Create a tiered risk scoring system to optimize loan terms instead of binary approve/deny decisions
- Partner with credit bureaus to enhance model with external data sources
- Build an early warning system that monitors existing customers for increasing default risk
- Design targeted intervention strategies for different risk segments to reduce default rates

## License
This project is licensed under the MIT License - see the LICENSE file for details.