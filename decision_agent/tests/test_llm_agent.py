
from prediction_agent.prediction_agent import PredictionAgent
from visual_agent.vision_agent import load_model, predict_waste
from decision_agent.llm_decision_agent import LLMDecisionAgent

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

# Initialize agents
prediction_agent = PredictionAgent()
decision_agent = LLMDecisionAgent()

vision_model_path = (
    ROOT / "prediction_agent" / "models"
    / "vision_agent_resnet18_dedup.pkl"
)

image_path = ROOT / "test_images" / "glass.jpg"

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

# Bin information
current_data = {
    "bin_id": "BIN_001",
    "fill_level": 78.0,
    "minutes": 30,
    "niveau_initial": 60.0
}

sensor_data = {
    "overflow_flag": False,
    "hazard_detected": False
}

# Agent 1: Prediction
prediction_result = prediction_agent.predict_bin(current_data)

# Agent 2: Visual
vision_model = load_model(str(vision_model_path))

waste_type, confidence = predict_waste(
    str(image_path),
    vision_model
)

visual_result = {
    "waste_type": waste_type,
    "confidence": confidence
}

# Agent 3: LLM Decision
decision_result = decision_agent.evaluate_bin(
    prediction_result,
    visual_result,
    sensor_data
)

# Display results
print("\n===== PREDICTION AGENT =====")
print(prediction_result)

print("\n===== VISUAL AGENT =====")
print(visual_result)

print("\n===== LLM DECISION AGENT =====")

for key, value in decision_result.items():
    print(f"{key}: {value}")
