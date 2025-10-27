# src/visualize_results.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc

def visualize_results():
    model = joblib.load('output/churn_model.pkl')
    scaler = joblib.load('output/scaler.pkl')

    df = pd.read_csv('data/churn_data.csv')
    X = df[['Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
    y = df['Churn']

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('output/confusion_matrix.png')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC Curve)')
    plt.legend(loc='lower right')
    plt.savefig('output/roc_curve.png')
    plt.show()

    print("✅ Visualizations saved in output/ folder.")

if __name__ == "__main__":
    visualize_results()
