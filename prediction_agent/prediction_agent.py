
import joblib
import pandas as pd


class PredictionAgent:

    def __init__(self, model_path):
        self.model = joblib.load(model_path)

        self.features = [
            "fill_level",
            "weight",
            "energy_used",
            "fill_rate",
            "weight_rate",
            "predicted_fill_10min"
        ]

    def predict_collection(self, current_data, previous_data):

        # Time difference in minutes
        time_diff = current_data["time_diff"]

        # Calculate fill rate
        fill_rate = (
            current_data["fill_level"] - previous_data["fill_level"]
        ) / time_diff

        # Calculate weight rate
        weight_rate = (
            current_data["weight"] - previous_data["weight"]
        ) / time_diff

        # Predict fill level after 10 minutes
        predicted_fill_10min = (
            current_data["fill_level"] + fill_rate * 10
        )

        # Prepare model input
        data = pd.DataFrame([{
            "fill_level": current_data["fill_level"],
            "weight": current_data["weight"],
            "energy_used": current_data["energy_used"],
            "fill_rate": fill_rate,
            "weight_rate": weight_rate,
            "predicted_fill_10min": predicted_fill_10min
        }])

        # Prediction
        prediction = self.model.predict(data)[0]

        # Result
        result = {
            "bin_id": current_data["bin_id"],
            "fill_level": current_data["fill_level"],
            "weight_rate": round(weight_rate, 4),
            "collection_required": int(prediction),
            "status": (
                "Collection Required"
                if prediction == 1
                else "No Collection Required"
            )
        }

        return result
