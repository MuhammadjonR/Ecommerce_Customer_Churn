# E-Commerce Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Predicting customer churn is crucial for e-commerce businesses aiming to retain valuable customers and reduce revenue loss. This project builds a complete machine learning pipeline — from raw data to a fully tuned production-ready model — including EDA, feature engineering, SMOTE, modeling, hyperparameter tuning, and SHAP explainability.

## 🚀 Project Pipeline

```
📂 Data → 🔍 EDA → 🧹 Preprocessing → 🛠 Feature Engineering →
⚖️ SMOTE → 🤖 Modeling → 🎯 Tuning → 📈 Evaluation → 📝 Explainability → 💾 Deployment-Ready Model
```

---


## 📊 Dataset Overview

The dataset contains customer behavior and engagement features such as:

- **Tenure**: Duration of customer relationship
- **DaySinceLastOrder**: Recency of last purchase
- **WarehouseToHome**: Distance from warehouse to customer
- **CashbackAmount**: Total cashback received
- **NumberOfDeviceRegistered**: Number of devices used
- **SatisfactionScore**: Customer satisfaction rating
- **Churn**: Target variable (0 = Active, 1 = Churned)

### Churn Distribution

```
0 (Active)   → 82.9%  
1 (Churned)  → 17.1%
```

⚠️ **Note**: The dataset is imbalanced, so SMOTE was applied during training to address class imbalance.

---

## 🔍 Exploratory Data Analysis

Key steps performed:

- Missing value inspection
- Outlier detection using IQR method
- Distribution plots & boxplots
- Correlation heatmap analysis
- Churn vs numerical & categorical features
- Feature relationships and insights

### Example Outlier Findings

```
Tenure:                      4 outliers
NumberOfDeviceRegistered:    271 outliers
DaySinceLastOrder:           223 outliers
CashbackAmount:              316 outliers
```

---

## 🧹 Preprocessing & Feature Engineering

### Numerical Features
- Standard scaling
- Outlier clipping
- Feature generation (e.g., `ActivityRatio`, `RecencyLevel`)

### Categorical Features
- One-hot encoding

### Pipeline
A reproducible `ColumnTransformer + Pipeline` structure ensures consistency between training and inference.

---

## ⚖️ Handling Imbalanced Data (SMOTE)

SMOTE (Synthetic Minority Over-sampling Technique) was applied **only to the training set** to balance class distribution:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train)
```

---

## 🤖 Models Trained & Compared

| Model                | ROC-AUC |
|----------------------|---------|
| Logistic Regression  | 0.894   |
| Decision Tree        | 0.844   |
| Random Forest        | 0.958   |
| Gradient Boosting    | 0.930   |
| SVM                  | 0.905   |
| **XGBoost (best)**   | **0.9597** |

🏆 **XGBoost** emerged as the best model with the highest ROC-AUC score.

---

## 🎯 Hyperparameter Tuning (Grid Search)

Grid search was performed to optimize XGBoost hyperparameters.

### Best Parameters Found

```python
{
    'model__subsample': 0.9,
    'model__n_estimators': 200,
    'model__min_child_weight': 1,
    'model__max_depth': 6,
    'model__learning_rate': 0.2,
    'model__gamma': 0,
    'model__colsample_bytree': 0.9
}
```

---

## 📈 Final Model Performance

### Metrics

- **ROC-AUC**: 0.9623

### Classification Report

```
              precision    recall  f1-score   support

           0       0.97      0.97      0.97       654
           1       0.86      0.83      0.85       135

    accuracy                           0.95       789
   macro avg       0.91      0.90      0.91       789
weighted avg       0.95      0.95      0.95       789
```

### Confusion Matrix

```
[[636  18]
 [ 23 112]]
```

✅ The model achieves **97% precision** on active customers and **83% recall** on churned customers.

---

## 📝 Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was used to understand feature influence on model predictions.

### Saving SHAP Plots

```python
import shap
import matplotlib.pyplot as plt

fig = plt.gcf()
fig.savefig("plots/shap_summary_plot.png", dpi=300, bbox_inches="tight")
plt.close(fig)
```

SHAP provides:
- **Global feature importance**: Which features matter most overall
- **Summary plots**: Distribution of feature impacts
- **Insights into churn drivers**: Understanding what causes customers to leave

---

## 💾 Saving the Model

The final model and preprocessing pipeline are saved for deployment:

```python
import joblib

joblib.dump(best_model, "models/final_xgboost_model.pkl")
joblib.dump(preprocessor, "models/xgb_preprocessing_pipeline.pkl")
```

---

## 🧠 Key Business Insights

1. **High recency (DaySinceLastOrder)** is the strongest churn indicator
   - Customers who haven't ordered recently are at highest risk

2. **Higher cashback leads to lower churn**
   - Incentive programs significantly impact retention

3. **Tenure strongly influences retention**
   - Newer customers churn more frequently

4. **Device registrations show nonlinear effects**
   - Multi-device users exhibit different behavior patterns

5. **XGBoost provides the best balance**
   - Optimal trade-off between accuracy and interpretability

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn joblib
```

### Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ecommerce-churn-prediction.git
   cd ecommerce-churn-prediction
   ```

2. **Run the notebooks**
   - Start with `01_EDA.ipynb` for data exploration
   - Continue with `02_Modelling.ipynb` for model training

3. **Load the saved model**
   ```python
   import joblib
   
   model = joblib.load("models/final_xgboost_model.pkl")
   preprocessor = joblib.load("models/xgb_preprocessing_pipeline.pkl")
   
   # Make predictions
   predictions = model.predict(preprocessor.transform(new_data))
   ```

---

## 📌 Next Steps (Optional Enhancements)

- [ ] Deploy via **FastAPI** or **Flask**
- [ ] Build a **Streamlit dashboard** for interactive predictions
- [ ] Add **automated retraining pipelines**
- [ ] Integrate into a **CRM system** for real-time churn alerts
- [ ] Implement **A/B testing** for retention strategies
- [ ] Add **monitoring and drift detection**

---

## ⭐ Contributions

Pull requests, issues, and suggestions are always welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- Dataset: [Source if applicable]
- Inspired by various e-commerce churn prediction research
- SHAP library for model interpretability

---

**⭐ If you find this project useful, please consider giving it a star!**
