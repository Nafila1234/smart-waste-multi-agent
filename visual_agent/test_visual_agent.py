
from pathlib import Path
from visual_agent.vision_agent import load_model, predict_waste

# Existing model location
project_root = Path(__file__).resolve().parent.parent

weights_path = (
    project_root
    / "prediction_agent"
    / "models"
    / "vision_agent_resnet18_dedup.pkl"
)

# Replace this with the path to an actual waste image
image_path = project_root / "test_images" / "glass.jpg"

# Load trained CNN
model = load_model(str(weights_path))

# Classify image
waste_type, confidence = predict_waste(str(image_path), model)

print("\n===== VISUAL AGENT RESULTS =====")
print("Waste type:", waste_type)
print("Confidence:", round(confidence, 4))
