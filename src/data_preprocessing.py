# src/data_preprocessing.py
import pandas as pd
import numpy as np

def generate_customer_data(n=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        'CustomerID': range(1, n + 1),
        'Age': np.random.randint(18, 70, n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Tenure': np.random.randint(1, 10, n),
        'Balance': np.random.uniform(0, 20000, n),
        'NumOfProducts': np.random.randint(1, 4, n),
        'HasCrCard': np.random.choice([0, 1], n),
        'IsActiveMember': np.random.choice([0, 1], n),
        'EstimatedSalary': np.random.uniform(20000, 100000, n)
    })

    # Random churn based on some rules
    data['Churn'] = np.where(
        (data['Balance'] < 5000) & (data['IsActiveMember'] == 0) & (data['Tenure'] < 3),
        1, 0
    )
    return data

def save_data(df, path='data/churn_data.csv'):
    df.to_csv(path, index=False)
    print(f"✅ Dataset saved to {path}")

if __name__ == "__main__":
    df = generate_customer_data()
    save_data(df)
