
from pathlib import Path

from prediction_agent.prediction_agent import PredictionAgent
from visual_agent.vision_agent import load_model, predict_waste
from decision_agent.decision_agent import DecisionAgent


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISION_MODEL_PATH = (
    PROJECT_ROOT
    / "prediction_agent"
    / "models"
    / "vision_agent_resnet18_dedup.pkl"
)

IMAGE_PATH = PROJECT_ROOT / "test_images" / "glass.jpg"


# ==========================================
# 2. CHECK REQUIRED FILES
# ==========================================

if not VISION_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Vision model not found: {VISION_MODEL_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Test image not found: {IMAGE_PATH}"
    )


# ==========================================
# 3. INITIALIZE THE THREE AGENTS
# ==========================================

prediction_agent = PredictionAgent()
decision_agent = DecisionAgent()

vision_model = load_model(str(VISION_MODEL_PATH))


# ==========================================
# 4. BIN INPUT DATA
# ==========================================

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


# ==========================================
# 5. AGENT 1: PREDICTION
# ==========================================

prediction_result = prediction_agent.predict_bin(
    current_data
)


# ==========================================
# 6. AGENT 2: VISUAL CLASSIFICATION
# ==========================================

waste_type, confidence = predict_waste(
    str(IMAGE_PATH),
    vision_model
)

visual_result = {
    "waste_type": waste_type,
    "confidence": confidence
}


# ==========================================
# 7. AGENT 3: INTELLIGENT DECISION
# ==========================================

decision_result = decision_agent.evaluate_bin(
    prediction_result,
    visual_result,
    sensor_data
)


# ==========================================
# 8. DISPLAY COMPLETE RESULTS
# ==========================================

print("\n========== AGENT 1: PREDICTION ==========")
for key, value in prediction_result.items():
    print(f"{key}: {value}")

print("\n========== AGENT 2: VISUAL ==========")
for key, value in visual_result.items():
    print(f"{key}: {value}")

print("\n========== AGENT 3: DECISION ==========")
for key, value in decision_result.items():
    print(f"{key}: {value}")

print("\n========== INTEGRATION STATUS ==========")
print("All three agents executed successfully.")
