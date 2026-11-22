
from prediction_agent.prediction_agent import PredictionAgent


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

# Initialize the Prediction Agent
agent = PredictionAgent()

# Run both models
result = agent.predict_bin(current_data, previous_data)

# Display results
print("\n===== PREDICTION AGENT RESULTS =====")

for key, value in result.items():
    print(f"{key}: {value}")
