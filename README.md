#  Customer Churn Prediction

This project focuses on **predicting customer churn** (whether a customer will leave or stay) using **machine learning** techniques.  
It uses simulated customer data and logistic regression to build a predictive model and evaluate its performance through various metrics and visualizations.

---

##  Project Structure

customer-churn-prediction/
│
├── data/ # Generated dataset (churn_data.csv)
├── output/ # Saved models and visualizations
│ ├── churn_model.pkl
│ ├── scaler.pkl
│ ├── confusion_matrix.png
│ └── roc_curve.png
├── src/
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ └── visualize_results.py
├── main.py
├── requirements.txt
└── README.md


---

##  Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
   cd customer-churn-prediction

pip install -r requirements.txt

python main.py

  Features

Synthetic customer data generation

Logistic Regression model training

Model evaluation with accuracy, precision, recall, F1-score

  Visualization:

Confusion Matrix

ROC Curve (AUC score)

Modular and reusable code structure


  Model Performance

Metric	Score
Accuracy	0.99
Precision	0.99
Recall	0.89
F1-score	0.93



