
from prediction_agent.prediction_agent import PredictionAgent
from visual_agent.vision_agent import load_model, predict_waste
from decision_agent.llm_decision_agent import LLMDecisionAgent
from routing_agent.routing_module import RoutingAgent
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


# ===== ROUTING MODULE =====

print("\n ROUTING MODULE RESULTS \n")

# Add geographical coordinates to the LLM decision
decision_with_location = {
    **decision_result,
    "latitude": 36.7665,
    "longitude": 3.4772
}

# Initialize routing module
routing_module = RoutingAgent(
    depot_latitude=36.7660,
    depot_longitude=3.4720
)

print("Depot:", routing_module.depot)
print("Bin coordinates:", decision_with_location["latitude"],
      decision_with_location["longitude"])

# Generate route using the actual LLM decision
route_result = routing_module.plan_route(
    [decision_with_location]
)

# Display route
for index, stop in enumerate(route_result["route"], start=1):
    print(f"\nStop {index}")
    print(f"Bin: {stop['bin_id']}")
    print(f"Priority: {stop['priority']}")
    print(f"Action: {stop['action']}")
    print(
        f"Distance: "
        f"{stop['distance_from_previous_km']} km"
    )

print("\n===== ROUTE SUMMARY =====")
print(f"Total distance: {route_result['total_distance_km']} km")
print(f"Return distance: {route_result['return_distance_km']} km")
print(route_result["message"])

# Display results
print("\n===== PREDICTION AGENT =====")
print(prediction_result)

print("\n===== VISUAL AGENT =====")
print(visual_result)

print("\n===== LLM DECISION AGENT =====")

for key, value in decision_result.items():
    print(f"{key}: {value}")
