# main.py
from src.data_preprocessing import generate_customer_data, save_data
from src.model_training import train_churn_model
from src.model_evaluation import evaluate_model
from src.visualize_results import visualize_results

def main():
    print(" Generating dataset...")
    df = generate_customer_data()
    save_data(df)

    print(" Training model...")
    train_churn_model()

    print(" Evaluating model...")
    evaluate_model()

    print(" Visualizing results...")
    visualize_results()

if __name__ == "__main__":
    main()
