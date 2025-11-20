📦 E-Commerce Customer Churn Prediction

This project builds a complete end-to-end machine learning pipeline to predict churn in an e-commerce business.
It includes EDA, outlier analysis, feature engineering, SMOTE balancing, model training, hyperparameter tuning, and SHAP explainability.



📊 Dataset Overview

The dataset contains customer behavioral, demographic, and purchasing features such as:

Tenure

WarehouseToHome

NumberOfDeviceRegistered

DaySinceLastOrder

CashbackAmount

SatisfactionScore

Churn (target)

Churn Distribution
0 → 82.9%  
1 → 17.1%


The dataset is highly imbalanced, so SMOTE was applied during model training.

🔍 Exploratory Data Analysis (EDA)

Key analysis steps:

Missing value inspection

Outlier detection using IQR

Distribution plots and boxplots

Correlation heatmap

Churn vs numerical & categorical features

Business insights discovery

Outlier Summary
Tenure: 4  
WarehouseToHome: 1  
NumberOfDeviceRegistered: 271  
DaySinceLastOrder: 223  
CashbackAmount: 316  

🧹 Preprocessing & Feature Engineering

Performed using ColumnTransformer + Pipeline:

Numerical

IQR outlier clipping

Standard scaling

Categorical

One-Hot Encoding

Feature Engineering

Created meaningful features such as:

RecencyLevel

ActivityRatio

CashbackEfficiency

All transformations are included inside a reproducible pipeline.

⚖️ Handling Class Imbalance (SMOTE)

Applied only to the training set to prevent data leakage:

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train)

🤖 Models Trained & Compared
Model	ROC-AUC
Logistic Regression	0.894
Decision Tree	0.844
Random Forest	0.958
Gradient Boosting	0.930
SVM	0.905
XGBoost (best)	0.9597
🎛 Hyperparameter Tuning (GridSearchCV)

Best XGBoost parameters:

{
 'model__subsample': 0.9,
 'model__n_estimators': 200,
 'model__min_child_weight': 1,
 'model__max_depth': 6,
 'model__learning_rate': 0.2,
 'model__gamma': 0,
 'model__colsample_bytree': 0.9
}

📈 Final Model Performance
ROC-AUC: 0.9623
Classification Report
precision    recall  f1-score   support

0       0.97      0.97      0.97       654
1       0.86      0.83      0.85       135

Confusion Matrix
[[636  18]
 [ 23 112]]

📝 Explainability (SHAP)

A complete SHAP analysis was performed to understand feature contributions.

The summary plot is saved at:

plots/shap_summary_plot.png


Example code used:

fig = plt.gcf()
fig.savefig("plots/shap_summary_plot.png", dpi=300, bbox_inches="tight")
plt.close(fig)


SHAP helps identify:

Most important churn drivers

Feature effect direction

Customer-level explanations

💾 Saving the Model
import joblib

joblib.dump(best_model, "models/final_xgboost_model.pkl")
joblib.dump(preprocessor, "models/xgb_preprocessing_pipeline.pkl")

📌 Key Business Insights

Customers with high recency (DaySinceLastOrder) are far more likely to churn.

Higher cashback reduces churn likelihood.

Low tenure customers churn at significantly higher rates.

Device registration behavior has nonlinear patterns captured well by XGBoost.

🚀 Next Steps (Future Work)

Add FastAPI model deployment

Build a Streamlit dashboard

Perform A/B testing for retention strategies

Integrate model outputs into CRM systems
