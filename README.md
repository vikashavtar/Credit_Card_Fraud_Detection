# Credit Card Fraud Detection

## Overview
This project involves building a model to detect fraudulent credit card transactions. The dataset is highly imbalanced, with fraudulent transactions constituting only 0.17% of the dataset. This project demonstrates various techniques to address the imbalance and create an effective fraud detection system.

## Goals
1. Build a credit card fraud detection model.
2. Address the challenges of working with imbalanced datasets.
3. Explore and evaluate multiple machine learning models.
4. Apply techniques such as anomaly detection and semi-supervised classification.

---

## Table of Contents
1. [Understanding the Data](#understanding-the-data)
2. [Preprocessing](#preprocessing)
3. [Handling Imbalance](#handling-imbalance)
4. [Dimensionality Reduction](#dimensionality-reduction)
5. [Model Evaluation](#model-evaluation)

---

## Understanding the Data
- **Dataset**: The features are anonymized, and the dataset includes time, transaction amount, and class labels.
- **Imbalance**: Only 0.17% of transactions are fraudulent.
- **Exploration**: Used descriptive statistics and visualizations (scatter plots, count plots, distribution plots) to understand the dataset.

## Preprocessing
1. **Scaling**: Time and Amount columns were scaled using RobustScaler.
2. **Splitting**: Created train and test splits ensuring balanced class representation.
3. **Data Imbalance Handling**: Techniques like SMOTE (oversampling) and NearMiss (undersampling) were applied to address the imbalance.

## Handling Imbalance
### Undersampling
- Balanced the dataset by reducing non-fraudulent transactions to match the number of fraudulent ones.
- Visualized class distribution before and after.
- Evaluated impact of undersampling on the model.

### Oversampling
- SMOTE (Synthetic Minority Oversampling Technique) was used to generate synthetic samples for the minority class.

## Dimensionality Reduction
- **Techniques**: PCA, t-SNE, and Truncated SVD were evaluated.
- **Selected**: t-SNE provided the best separation of fraudulent and non-fraudulent clusters.
- **Visualization**: Scatter plots of clusters from each technique were compared.

## Model Evaluation
### Classifiers
1. **Logistic Regression**:
   - Tuned hyperparameters using GridSearchCV.
   - Achieved high precision and recall on imbalanced data.

2. **XGBoost**:
   - Evaluated using cross-validation.
   - Slightly outperformed Logistic Regression in overall performance.

### Metrics
- Precision, Recall, F1-score, Accuracy, ROC-AUC.
- Precision-recall curves and learning curves were used to analyze model performance.

### Observations
- Logistic Regression ROC-AUC: 0.98
- XGBoost ROC-AUC: 0.94

### Visualizations
- Plotted learning curves to detect overfitting.
- ROC and precision-recall curves to evaluate model predictions.

---

## To-Do
1. Add more visualizations to enhance exploratory data analysis.
2. Experiment with different classifiers and hyperparameters.
3. Incorporate cost-sensitive learning to further mitigate class imbalance.
4. Validate results with additional datasets.

---

## Conclusion
This project highlights the complexities of working with imbalanced datasets and demonstrates how various techniques, from preprocessing to model evaluation, can help build an effective fraud detection system. Future work will focus on refining the model and applying it to real-world scenarios.

---

## Requirements
- Python 3.7+
- Libraries: numpy, pandas, sklearn, imbalanced-learn, matplotlib, seaborn, tensorflow, xgboost

## How to Run
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Open and execute `Credit_Card_Fraud_Detection.ipynb` in Jupyter Notebook or Google Colab.

---

## License
MIT License

---

## Acknowledgments
- Dataset sourced from [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- Techniques inspired by academic papers and machine learning best practices.

