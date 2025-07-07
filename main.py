from src.data_preprocessing import load_data, preprocess_data
from src.model import train_model, save_model
from src.evaluate import evaluate_model, plot_confusion_matrix
from src.utils import set_seed, ensure_dir
import joblib


def main():
    # Step 0: Setup
    set_seed(42)
    ensure_dir("outputs/models")
    ensure_dir("outputs/plots")

    # Step 1: Load dataset
    data_path = "data/dataset-latest.csv"

    target_column = "Printable"
    df = load_data(data_path)

    # Step 2: Preprocess (Remarks is included during training internally)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, target_column, fit=True)

    # Step 3: Train model
    model = train_model(X_train, y_train)
    print("✅ Model training completed.")

    # Step 4: Evaluate
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Step 5: Save artifacts
    save_model(model, "outputs/models/printability_model.pkl")
    joblib.dump(preprocessor, "outputs/models/preprocessor.pkl")
    print("💾 Model and preprocessor saved in 'outputs/models/'")


if __name__ == "__main__":
    main()
