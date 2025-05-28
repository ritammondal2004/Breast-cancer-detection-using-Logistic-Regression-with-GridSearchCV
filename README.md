# Breast-cancer-detection-using-Logistic-Regression-with-GridSearchCV
detect whether the cancer is benign or malignant using Logistic regression, SVM building the best model using GridSearchCV

# 🧠 Breast Cancer Detection using Logistic Regression and SVC

This project focuses on building a supervised machine learning pipeline to predict whether a tumor is **benign (0)** or **malignant (1)** using the **Breast Cancer Wisconsin Dataset**. We use **Logistic Regression** and **Support Vector Machine (SVC)** with **hyperparameter tuning (GridSearchCV)** and visualizations to evaluate model performance.

---

## 📁 Dataset

- **Source**: scikit-learn’s built-in `load_breast_cancer()` dataset
- **Samples**: 569 tumor records
- **Features**: 30 numerical features (e.g., `radius_mean`, `concavity_mean`, `area_mean`, etc.)
- **Target**: `diagnosis` (Benign = 0, Malignant = 1)

---

## 🎯 Objective

- Explore, clean, and visualize the dataset
- Train ML models to classify tumors
- Tune hyperparameters using GridSearchCV
- Visualize performance (ROC, thresholds, confusion matrix)
- Allow user-defined inputs for real-time predictions

---

## 🧪 ML Approach

- **Preprocessing**:
  - Feature scaling using `StandardScaler`
  - Label encoding (`M` → 1, `B` → 0)
- **Models used**:
  - Logistic Regression
  - SVC (linear, rbf kernels)
- **Hyperparameter tuning**:
  - GridSearchCV with 5-fold CV
  - Tested multiple solvers (`liblinear`, `saga`) and penalties (`l1`, `l2`, `elasticnet`)
- **Final Model**:
  - Selected best model using `.best_estimator_`

---

## 📊 Evaluation Metrics


print("confusion matrix:\n", confusion_matrix(y_test, y_pred))

- Accuracy `accuracy_score(y_test,y_pred)*100:.2f`
- Precision `precision_score(y_test,y_pred)*100:.2f` & Recall `recall_score(y_test,y_pred)*100:.2f`
- Confusion Matrix (visualized as heatmap)
  `confusion_matrix(y_test, y_pred)`
  
  ```
  cm = confusion_matrix(y_test, y_pred)
  sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])  
- ROC Curve & AUC Score
- Threshold Tuning Trade-off Curve 

---

## 📈 Visualizations

- 📎 Diagnosis distribution pie chart
- 🧭 3D Scatter plot (`area_mean` vs `concavity_mean` vs `smoothness_mean`)
- 📉 Precision–Recall–Accuracy vs Threshold plot
- 🧊 Confusion matrix heatmap

---


## 👨‍💻 Author 
**Ritam Mondal**
- Dual Degree, Industrial & Systems Engineering
- Indian Institute of Technology Kharagpur
- GitHub · LinkedIn https://www.linkedin.com/in/ritam-mondal-86a369287/ 
- Email: ritamm134@gmail.com
