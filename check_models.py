
import joblib

models = {
    "Decision Tree": "prediction_agent/models/decision_tree_model.pkl",
    "Linear Regression": "prediction_agent/models/modele_regression_lineaire.pkl"
}

for name, path in models.items():
    print(f"\n--- {name} ---")

    model = joblib.load(path)

    print("Model loaded successfully!")
    print("Model type:", type(model).__name__)
    print("Expected features:", getattr(model, "feature_names_in_", "Not available"))
    print("Number of features:", getattr(model, "n_features_in_", "Not available"))
