# src/model_evaluation.py
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

def evaluate_model():
    model = joblib.load('output/churn_model.pkl')
    scaler = joblib.load('output/scaler.pkl')

    df = pd.read_csv('data/churn_data.csv')
    X = df[['Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
    y = df['Churn']

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    print("✅ Model Evaluation:")
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))
    print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")

if __name__ == "__main__":
    evaluate_model()
