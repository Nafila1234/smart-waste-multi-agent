
from prediction_agent.prediction_agent import PredictionAgent
from decision_agent.decision_agent import DecisionAgent


# Sample sensor data
current_data = {
    "bin_id": "BIN_001",
    "fill_level": 78.0,
    "weight": 25.0,
    "energy_used": 12.0,
    "time_diff": 10,
    "minutes": 30,
    "niveau_initial": 60.0
}

previous_data = {
    "fill_level": 75.0,
    "weight": 23.0
}

# Sample visual and sensor information
visual = {
    "waste_type": "Plastic",
    "hazard_detected": False
}

sensor = {
    "overflow_flag": False
}

# Initialize agents
prediction_agent = PredictionAgent()
decision_agent = DecisionAgent()

# Run Prediction Agent
prediction = prediction_agent.predict_bin(
    current_data,
    previous_data
)

# Pass prediction to Decision Agent
decision = decision_agent.evaluate_bin(
    prediction,
    visual,
    sensor
)

# Display results
print("\n===== PREDICTION RESULT =====")
print(prediction)

print("\n===== FINAL DECISION =====")
for key, value in decision.items():
    print(f"{key}: {value}")
