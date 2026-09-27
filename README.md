# Credit Card Fraud Detection MLOps Final Project

## Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) workflow for credit card fraud detection.

The system uses an XGBoost classification model to identify potentially fraudulent credit card transactions. The project includes data preprocessing, model training and evaluation, API deployment with FastAPI, Docker containerization, automated testing, monitoring components, and CI/CD using GitHub Actions.

## Model Performance

The final tuned XGBoost model achieved:

- Precision: 0.8696
- Recall: 0.8108
- F1-Score: 0.8392
- ROC-AUC: 0.9686

Final test confusion matrix:

- True Negatives: 42,639
- False Positives: 9
- False Negatives: 14
- True Positives: 60

## Project Structure

```text
Final-Project/
├── app/
│   └── main.py
├── models/
│   ├── xgboost_fraud_model.pkl
│   ├── scaler.pkl
│   ├── feature_info.pkl
│   └── final_metrics.pkl
├── monitoring/
│   └── monitor.py
├── notebooks/
│   └── fraud_detection.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── retrain.py
├── tests/
│   └── test_api.py
├── .github/workflows/
│   └── ci-cd.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Final_Report.pdf
└── README.md
