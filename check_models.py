from pathlib import Path
import joblib
import pandas as pd
import sklearn

MODELS_DIR = Path(__file__).resolve().parent / "prediction_agent" / "models"

# Must match what PredictionAgent builds
TREE_FEATURES = [
    "fill_level", "weight", "energy_used",
    "fill_rate", "weight_rate", "predicted_fill_10min",
]
REGRESSION_FEATURES = ["minutes", "niveau_initial"]

print("scikit-learn version:", sklearn.__version__)

def load(name):
    path = MODELS_DIR / name
    try:
        model = joblib.load(path)
    except FileNotFoundError:
        raise SystemExit(f"Missing model file: {path}")
    except Exception as e:
        raise SystemExit(f"Could not load {path}: {e}")
    print(f"\n--- {name} ---")
    print("Type:", type(model).__name__)
    print("Features:", list(getattr(model, "feature_names_in_", [])))
    return model

tree = load("decision_tree_model.pkl")
reg = load("modele_regression_lineaire.pkl")

# 1. Feature names must match what the agent sends
assert list(tree.feature_names_in_) == TREE_FEATURES, \
    f"Tree features mismatch: {list(tree.feature_names_in_)}"
assert list(reg.feature_names_in_) == REGRESSION_FEATURES, \
    f"Regression features mismatch: {list(reg.feature_names_in_)}"

# 2. Regression details used by the agent
print("\nRegression coef:", dict(zip(reg.feature_names_in_, reg.coef_)))
print("Regression intercept:", reg.intercept_)
assert reg.coef_[list(reg.feature_names_in_).index("minutes")] > 0, \
    "Slope on 'minutes' should be positive (bins fill up over time)"

# 3. Test predictions
tree_in = pd.DataFrame([[90, 40, 10, 0.3, 0.2, 95]], columns=TREE_FEATURES)
reg_in = pd.DataFrame([[40, 60]], columns=REGRESSION_FEATURES)
print("\nTree prediction (full bin):", tree.predict(tree_in)[0])
print("Regression prediction:", reg.predict(reg_in)[0])

print("\nAll checks passed.")