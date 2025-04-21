# Telecom_Customer_Churn_DL_Practical9
 DL Practical 9 

 # Telecom Customer Churn Detection using Deep Learning

## 🧠 Problem Statement

In the telecom industry, retaining existing customers is more cost-effective than acquiring new ones. This project aims to predict customer churn — i.e., whether a customer is likely to discontinue their telecom services — using deep learning models. By proactively identifying high-risk customers, businesses can implement retention strategies to reduce churn.

## 🔍 Explanation

This project implements and compares three deep learning-based approaches to predict customer churn:

1. **Deep Neural Network (DNN)**  
   A multi-layer feedforward neural network trained on scaled, encoded customer data to classify whether the customer will churn.

2. **Autoencoder for Anomaly Detection**  
   An unsupervised deep learning technique that learns to reconstruct normal customer data. Higher reconstruction errors suggest anomalies — potential churners.

3. **LSTM (Long Short-Term Memory)**  
   Although originally used for sequence modeling, LSTM is experimented with here to explore churn patterns over customer lifecycle (tenure-based input).

Key steps in the pipeline:
- Data preprocessing including handling categorical variables via one-hot encoding.
- Feature scaling using `StandardScaler`.
- Training separate models on 30 preprocessed features.
- Building a prediction app where users can select any of the three models to get churn probability or anomaly score.

## 📂 Dataset Link

Dataset used: [Telco Customer Churn Dataset - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---
## Deployment

Locally deployed using localhost and streamlit by making use of saved h5 models
