
import joblib
import pandas as pd
from pathlib import Path


class PredictionAgent:

    def __init__(self):

        # Locate models folder
        models_dir = Path(__file__).resolve().parent / "models"

        # Load trained models
        self.decision_tree = joblib.load(
            models_dir / "decision_tree_model.pkl"
        )

        self.linear_regression = joblib.load(
            models_dir / "modele_regression_lineaire.pkl"
        )

        self.collection_threshold = 85

        # Decision Tree now uses only fill_level
        self.tree_features = ["fill_level"]

        # Check Linear Regression input features
        self.regression_features = [
            "minutes",
            "niveau_initial"
        ]

    def predict_bin(self, current_data, previous_data=None):

        # 1. Validate required input
        required_fields = [
            "bin_id",
            "fill_level",
            "minutes",
            "niveau_initial"
        ]

        for field in required_fields:
            if field not in current_data:
                raise ValueError(
                    f"Missing required field: {field}"
                )

        current_fill = float(current_data["fill_level"])

        if not 0 <= current_fill <= 100:
            raise ValueError(
                "fill_level must be between 0 and 100"
            )

        # 2. Decision Tree: current collection prediction
        tree_input = pd.DataFrame(
            [{"fill_level": current_fill}],
            columns=self.tree_features
        )

        tree_prediction = int(
            self.decision_tree.predict(tree_input)[0]
        )

        # 3. Linear Regression: predict fill after 10 minutes
        future_minutes = float(current_data["minutes"]) + 10

        regression_input = pd.DataFrame(
            [{
                "minutes": future_minutes,
                "niveau_initial": float(
                    current_data["niveau_initial"]
                )
            }],
            columns=self.regression_features
        )

        raw_predicted_fill = float(
            self.linear_regression.predict(regression_input)[0]
        )

        predicted_fill_10min = max(
            0.0,
            min(100.0, raw_predicted_fill)
        )

        # 4. Determine whether future collection is required
        regression_prediction = (
            predicted_fill_10min >= self.collection_threshold
        )

        # 5. Compare model predictions
        prediction_conflict = (
            bool(tree_prediction) != regression_prediction
        )

        # 6. Estimate remaining time using regression slope
        remaining_time = None

        if hasattr(self.linear_regression, "feature_names_in_"):

            feature_names = list(
                self.linear_regression.feature_names_in_
            )

            if "minutes" in feature_names:

                minutes_index = feature_names.index("minutes")

                filling_rate = float(
                    self.linear_regression.coef_[minutes_index]
                )

                if filling_rate > 0:

                    remaining_time = max(
                        0.0,
                        (
                            self.collection_threshold - current_fill
                        ) / filling_rate
                    )

        else:
            filling_rate = None

        # 7. Return combined prediction results
        return {
            "bin_id": current_data["bin_id"],

            "current_fill": current_fill,

            "predicted_fill_10min": round(
                predicted_fill_10min, 2
            ),

            "remaining_time_minutes": (
                round(float(remaining_time), 2)
                if remaining_time is not None
                else None
            ),

            "tree_collection_prediction": tree_prediction,

            "tree_status": (
                "Collection Required"
                if tree_prediction == 1
                else "No Collection Required"
            ),

            "regression_collection_prediction": regression_prediction,

            "prediction_conflict": prediction_conflict
        }
