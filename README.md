# Credit-Card-Fraud-Detection-using-ML
Credit Card Detection
This project focuses on detecting fraudulent credit card transactions using machine learning algorithms. The dataset used for this project is from Kaggle and contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly unbalanced, with 492 frauds out of 284,807 transactions (0.172%).

# Table of Contents
Introduction Dataset Preprocessing Modeling Evaluation Conclusion Installation Usage Contributing License

# Introduction
The primary goal of this project is to build a robust model to accurately detect fraudulent credit card transactions. We have implemented and compared three different machine learning algorithms: Logistic Regression, Support Vector Classifier (SVC), and Random Forest Classifier.

# Dataset
The dataset used in this project is available on Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Number of transactions: 284,807 Number of fraudulent transactions: 492 Features: 30 numerical features (V1-V28, Amount, Time) Target: 1 for fraudulent transactions, 0 for non-fraudulent transactions Preprocessing Data Loading: Load the dataset using pandas. Data Cleaning: Check for missing values and handle them accordingly (though this dataset has no missing values). Feature Scaling: Normalize the 'Amount' and 'Time' features using StandardScaler. Splitting the Data: Split the dataset into training and testing sets using train_test_split. Modeling

# Logistic Regression:
Implemented Logistic Regression using scikit-learn. Tuned hyperparameters using GridSearchCV.

# Support Vector Classifier (SVC):
Implemented SVC using scikit-learn. Used GridSearchCV for hyperparameter tuning.

# Random Forest Classifier:
Implemented Random Forest using scikit-learn. Performed hyperparameter tuning using GridSearchCV.

# Evaluation
Confusion Matrix: Evaluate the performance using confusion matrix. Classification Report: Generate classification report including precision, recall, f1-score. ROC Curve: Plot ROC curve and calculate AUC for each model. Model Comparison: Compare the performance of all three models and select the best one based on evaluation metrics.

Conclusion
The project demonstrates the use of different machine learning algorithms to detect credit card fraud. The Random Forest Classifier provided the best results in terms of precision, recall, and AUC. Future work can include trying out other advanced algorithms and techniques to further improve the model performance.
