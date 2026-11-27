from prediction_agent.prediction_agent import PredictionAgent

agent = PredictionAgent()

current_data = {
    "bin_id": "BIN_001",
    "fill_level": 78.0,
    "minutes": 30,
    "niveau_initial": 60.0
}

result = agent.predict_bin(current_data)

print("PREDICTION RESULTS")
for key, value in result.items():
    print(f"{key}: {value}")