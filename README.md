##E-Commerce Customer Churn Prediction

This project builds a machine-learning pipeline to predict whether an e-commerce customer will churn.
The workflow includes exploratory data analysis (EDA), feature engineering, outlier analysis, SMOTE balancing, model training, hyperparameter tuning, and model explainability using SHAP.
E-commerce companies lose a large portion of customers due to churn.
The goal of this project is to:

Identify which customers are at high risk of churning

Understand drivers of churn using model explainability tools

Provide a production-ready machine learning model

High recency (DaySinceLastOrder) is the strongest churn indicator.

Higher cashback leads to lower churn → incentive programs matter.

Tenure strongly influences retention — newer customers churn more.

Device registrations show nonlinear effects.

XGBoost gives the best balance between accuracy and interpretability.
