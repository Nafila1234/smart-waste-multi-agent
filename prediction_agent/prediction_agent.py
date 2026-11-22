
import joblib
import pandas as pd
from pathlib import Path


class PredictionAgent:

    def __init__(self):

        # Locate the models folder
        models_dir = Path(__file__).resolve().parent / "models"

        # Load both trained models
        self.decision_tree = joblib.load(
            models_dir / "decision_tree_model.pkl"
        )

        self.linear_regression = joblib.load(
            models_dir / "modele_regression_lineaire.pkl"
        )

        self.collection_threshold = 85

        # Features expected by Decision Tree
        self.tree_features = [
            "fill_level",
            "weight",
            "energy_used",
            "fill_rate",
            "weight_rate",
            "predicted_fill_10min"
        ]

    def predict_bin(self, current_data, previous_data):

        # 1. Calculate filling rates
        time_diff = current_data["time_diff"]

        if time_diff <= 0:
            raise ValueError("time_diff must be greater than zero")

        fill_rate = (
            current_data["fill_level"] -
            previous_data["fill_level"]
        ) / time_diff

        weight_rate = (
            current_data["weight"] -
            previous_data["weight"]
        ) / time_diff

        # 2. Predict fill level after 10 minutes
        future_minutes = current_data["minutes"] + 10

        regression_input = pd.DataFrame([{
            "minutes": future_minutes,
            "niveau_initial": current_data["niveau_initial"]
        }])

        raw_predicted_fill_10min = float(
            self.linear_regression.predict(regression_input)[0]
        )
        predicted_fill_10min = max(
            0.0,
            min(100.0, raw_predicted_fill_10min)
        )

        # 3. Estimate remaining time using regression slope
        feature_names = list(
            self.linear_regression.feature_names_in_
        )

        minutes_index = feature_names.index("minutes")
        filling_rate = self.linear_regression.coef_[minutes_index]

        if filling_rate > 0:
            remaining_time = (
                self.collection_threshold -
                current_data["fill_level"]
            ) / filling_rate

            remaining_time = max(0, remaining_time)
        else:
            remaining_time = None

        # 4. Prepare Decision Tree input
        tree_input = pd.DataFrame([{
            "fill_level": current_data["fill_level"],
            "weight": current_data["weight"],
            "energy_used": current_data["energy_used"],
            "fill_rate": fill_rate,
            "weight_rate": weight_rate,
            "predicted_fill_10min": predicted_fill_10min
        }], columns=self.tree_features)

        # 5. Predict collection requirement
        tree_prediction = int(
            self.decision_tree.predict(tree_input)[0]
        )

        # 6. Compare both model signals
        regression_prediction = (
            predicted_fill_10min >= self.collection_threshold
        )

        prediction_conflict = (
            bool(tree_prediction) != regression_prediction
        )

        # 7. Return combined results
        return {
            "bin_id": current_data["bin_id"],
            "current_fill": current_data["fill_level"],
            "predicted_fill_10min": round(
                float(predicted_fill_10min), 2
            ),
            "remaining_time_minutes": (
                round(float(remaining_time), 2)
                if remaining_time is not None else None
            ),
            "filling_rate": round(float(filling_rate), 4),
            "tree_collection_prediction": tree_prediction,
            "tree_status": (
                "Collection Required"
                if tree_prediction == 1
                else "No Collection Required"
            ),
            "regression_collection_prediction": regression_prediction,
            "prediction_conflict": prediction_conflict
        }
