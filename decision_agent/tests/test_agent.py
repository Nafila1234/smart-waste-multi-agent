
from decision_agent.decision_agent import DecisionAgent
from prediction_agent.prediction_agent import PredictionAgent


# Initialize agents
prediction_agent = PredictionAgent()
decision_agent = DecisionAgent()


# Example bin information
current_data = {
    "bin_id": "BIN_001",
    "fill_level": 78.0,
    "minutes": 30,
    "niveau_initial": 60.0
}

# Sensor information (simulated for this test)
sensor_data = {
    "overflow_flag": False,
    "hazard_detected": False
}

# Visual information (simulated for this test)
visual_data = {
    "waste_type": "glass",
    "confidence": 0.89
}


# Run Prediction Agent
prediction_result = prediction_agent.predict_bin(current_data)

print("\n===== PREDICTION AGENT =====")
print(prediction_result)


# Run Decision Agent
decision_result = decision_agent.evaluate_bin(
    prediction_result,
    visual_data,
    sensor_data
)

print("\n===== DECISION AGENT =====")

for key, value in decision_result.items():
    print(f"{key}: {value}")
