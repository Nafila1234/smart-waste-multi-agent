
from pathlib import Path

from prediction_agent.prediction_agent import PredictionAgent
from decision_agent.decision_agent import DecisionAgent
from visual_agent.vision_agent import load_model, predict_waste


# ==========================================
# 1. FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Update this when your teammate provides the trained weights.
MODEL_PATH = BASE_DIR / "visual_agent" / "vision_agent_resnet18_dedup.pth"

# Choose an actual image from your test dataset.
IMAGE_PATH = BASE_DIR / "data" / "test_waste.jpg"


# ==========================================
# 2. SAMPLE SENSOR DATA
# ==========================================

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

sensor = {
    "overflow_flag": False
}


# ==========================================
# 3. INITIALIZE AGENTS
# ==========================================

prediction_agent = PredictionAgent()
decision_agent = DecisionAgent()


# ==========================================
# 4. PREDICTION AGENT
# ==========================================

prediction = prediction_agent.predict_bin(
    current_data,
    previous_data
)


# ==========================================
# 5. VISUAL AGENT
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model weights not found: {MODEL_PATH}\n"
        "Ask your teammate for the trained .pth checkpoint."
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Test image not found: {IMAGE_PATH}\n"
        "Choose an existing image from your dataset."
    )

# Load the teammate's trained model
visual_model = load_model(str(MODEL_PATH))

# Classify the image
waste_type, confidence = predict_waste(
    str(IMAGE_PATH),
    visual_model
)

# Format the output for the Decision Agent
visual = {
    "waste_type": waste_type,
    "confidence": confidence,
    "hazard_detected": False
}


# ==========================================
# 6. DECISION AGENT
# ==========================================

decision = decision_agent.evaluate_bin(
    prediction,
    visual,
    sensor
)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n========== PREDICTION AGENT ==========")

for key, value in prediction.items():
    print(f"{key}: {value}")


print("\n========== VISUAL AGENT ==========")

for key, value in visual.items():
    print(f"{key}: {value}")


print("\n========== SENSOR INFORMATION ==========")

for key, value in sensor.items():
    print(f"{key}: {value}")


print("\n========== FINAL DECISION ==========")

for key, value in decision.items():
    print(f"{key}: {value}")
