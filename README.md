# 💳 Credit Card Fraud Detection

This project demonstrates how to detect fraudulent credit card transactions using machine learning techniques. The dataset used is highly imbalanced, and appropriate resampling strategies are employed to address this issue. The goal is to accurately classify transactions as fraudulent or legitimate.

## 📌 Objective

To build a predictive model that can effectively identify fraudulent credit card transactions, helping financial institutions minimize loss and improve security.

---

## 📂 Dataset

- The dataset consists of credit card transactions labeled as fraudulent (1) or legitimate (0).
- It is preprocessed and transformed with PCA (Principal Component Analysis) for privacy and dimensionality reduction.
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## ⚙️ Tools and Libraries Used

- Python
- Pandas
- Numpy
- Matplotlib / Seaborn
- Scikit-learn (for modeling, preprocessing, evaluation)
- Imbalanced-learn (for SMOTE)

---

## 🧪 Methodology

1. **Data Exploration and Preprocessing**
   - Load and inspect the data.
   - Analyze class distribution and scale the features.
   - Split into train and test sets.

2. **Handling Imbalanced Data**
   - Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes in the training set.

3. **Modeling**
   - Train a **Random Forest Classifier** on the balanced data.
   - Predict on the test set.

4. **Evaluation Metrics**
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-score)
   - ROC-AUC Score

---

## ✅ Results

The model is evaluated primarily based on **Recall** to minimize false negatives (i.e., missed frauds), which is more critical in fraud detection scenarios.

- The balanced data improves model sensitivity.
- Random Forest yields a good trade-off between performance and interpretability.

---

## 📊 Visualization

Graphs and charts (confusion matrix, ROC curves) help interpret model performance and data distribution. (Add plots if available.)

---

## 🚀 Future Improvements

- Experiment with advanced models (XGBoost, LightGBM).
- Apply anomaly detection techniques.
- Incorporate real-time detection pipelines.

---

## 👩‍💻 Author

Harsheeta Khandelwal  
[LinkedIn](https://www.linkedin.com/in/harsheeta-khandelwal/)  
Email: [harsheetakhandelwal698@gmail.com]


