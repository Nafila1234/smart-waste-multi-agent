
from decision_agent.config import FILL_THRESHOLD, PRIORITIES


class DecisionAgent:

    def evaluate_bin(self, prediction, visual, sensor):

        # 1. Collect information
        bin_id = prediction.get("bin_id")

        current_fill = prediction.get(
            "current_fill",
            prediction.get("fill_level", 0)
        )

        predicted_fill = prediction.get("predicted_fill_10min")

        tree_prediction = prediction.get(
            "tree_collection_prediction", 0
        )

        regression_prediction = prediction.get(
            "regression_collection_prediction", False
        )

        prediction_conflict = prediction.get(
            "prediction_conflict", False
        )

        waste_type = visual.get("waste_type", "Unknown")

        hazard_detected = visual.get("hazard_detected", False)
        overflow_detected = sensor.get("overflow_flag", False)

        # 2. Initialize decision
        collection_required = False
        priority = "LOW"
        reason = "Bin does not require immediate collection."
        action = "MONITOR"

        # 3. Apply decision rules

        if hazard_detected:
            collection_required = True
            priority = "CRITICAL"
            reason = "Hazardous waste detected."
            action = "IMMEDIATE_COLLECTION"

        elif overflow_detected:
            collection_required = True
            priority = "CRITICAL"
            reason = "Bin overflow detected."
            action = "IMMEDIATE_COLLECTION"

        elif current_fill >= FILL_THRESHOLD:
            collection_required = True
            priority = "HIGH"
            reason = "Current fill level exceeds the threshold."
            action = "COLLECT"

        elif tree_prediction == 1 and regression_prediction:
            collection_required = True
            priority = "HIGH"
            reason = "Both models predict collection is required."
            action = "SCHEDULE_COLLECTION"

        elif tree_prediction == 1:
            collection_required = True
            priority = "MEDIUM"
            reason = "Decision Tree predicts collection is required."
            action = "SCHEDULE_COLLECTION"

        elif regression_prediction:
            collection_required = True
            priority = "MEDIUM"
            reason = "Linear Regression predicts the bin will reach the threshold."
            action = "SCHEDULE_COLLECTION"

        # 4. Generate final result
        return {
            "bin_id": bin_id,
            "current_fill": current_fill,
            "predicted_fill": predicted_fill,
            "waste_type": waste_type,
            "hazard_detected": hazard_detected,
            "overflow_detected": overflow_detected,
            "collection_required": collection_required,
            "priority": priority,
            "priority_value": PRIORITIES[priority],
            "action": action,
            "tree_prediction": tree_prediction,
            "regression_prediction": regression_prediction,
            "prediction_conflict": prediction_conflict,
            "reason": reason
        }
