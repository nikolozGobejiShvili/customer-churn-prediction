# src/model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

def train_churn_model(path='data/churn_data.csv'):
    df = pd.read_csv(path)

    X = df[['Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, 'output/churn_model.pkl')
    joblib.dump(scaler, 'output/scaler.pkl')

    print("✅ Model trained and saved successfully!")
    return model, X_test_scaled, y_test

if __name__ == "__main__":
    train_churn_model()
