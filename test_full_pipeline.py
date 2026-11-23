from prediction_agent.prediction_agent import PredictionAgent
from decision_agent.decision_agent import DecisionAgent
from routing_agent.routing_agent import RoutingAgent

# 1. INITIALIZE THE AGENTS


prediction_agent = PredictionAgent()
decision_agent = DecisionAgent()

routing_agent = RoutingAgent(
    depot_latitude=36.7500,
    depot_longitude=3.0500
)

# 2. SIMULATED BIN DATA

bins_data = [

    {
        "bin_id": "BIN_001",
        "current": {
            "bin_id": "BIN_001",
            "fill_level": 78.0,
            "weight": 25.0,
            "energy_used": 12.0,
            "time_diff": 10,
            "minutes": 30,
            "niveau_initial": 60.0
        },
        "previous": {
            "fill_level": 75.0,
            "weight": 23.0
        },
        "visual": {
            "waste_type": "Plastic",
            "hazard_detected": False
        },
        "sensor": {
            "overflow_flag": False
        },
        "latitude": 36.7510,
        "longitude": 3.0520
    },

    {
        "bin_id": "BIN_002",
        "current": {
            "bin_id": "BIN_002",
            "fill_level": 92.0,
            "weight": 40.0,
            "energy_used": 15.0,
            "time_diff": 10,
            "minutes": 30,
            "niveau_initial": 70.0
        },
        "previous": {
            "fill_level": 89.0,
            "weight": 38.0
        },
        "visual": {
            "waste_type": "Organic",
            "hazard_detected": False
        },
        "sensor": {
            "overflow_flag": False
        },
        "latitude": 36.7540,
        "longitude": 3.0550
    },

    {
        "bin_id": "BIN_003",
        "current": {
            "bin_id": "BIN_003",
            "fill_level": 30.0,
            "weight": 10.0,
            "energy_used": 5.0,
            "time_diff": 10,
            "minutes": 0,
            "niveau_initial": 20.0
        },
        "previous": {
            "fill_level": 28.0,
            "weight": 9.0
        },
        "visual": {
            "waste_type": "Paper",
            "hazard_detected": False
        },
        "sensor": {
            "overflow_flag": False
        },
        "latitude": 36.7520,
        "longitude": 3.0510
    },

    {
        "bin_id": "BIN_004",
        "current": {
            "bin_id": "BIN_004",
            "fill_level": 88.0,
            "weight": 32.0,
            "energy_used": 10.0,
            "time_diff": 10,
            "minutes": 30,
            "niveau_initial": 65.0
        },
        "previous": {
            "fill_level": 85.0,
            "weight": 30.0
        },
        "visual": {
            "waste_type": "Glass",
            "hazard_detected": True
        },
        "sensor": {
            "overflow_flag": False
        },
        "latitude": 36.7550,
        "longitude": 3.0540
    }
]


# 3. PREDICTION + DECISION

final_decisions = []

print("\n========== INDIVIDUAL BIN ANALYSIS ==========\n")

for bin_data in bins_data:

    # Prediction Agent
    prediction = prediction_agent.predict_bin(
        bin_data["current"],
        bin_data["previous"]
    )

    # Decision Agent
    decision = decision_agent.evaluate_bin(
        prediction,
        bin_data["visual"],
        bin_data["sensor"]
    )

    # Add GPS coordinates for Routing Agent
    decision["latitude"] = bin_data["latitude"]
    decision["longitude"] = bin_data["longitude"]

    final_decisions.append(decision)

    print(f"\nBin: {decision['bin_id']}")
    print(f"Current fill: {decision['current_fill']}%")
    print(f"Predicted fill: {decision['predicted_fill']}%")
    print(f"Waste type: {decision['waste_type']}")
    print(f"Priority: {decision['priority']}")
    print(f"Collection required: {decision['collection_required']}")
    print(f"Action: {decision['action']}")
    print(f"Reason: {decision['reason']}")

# 4. ROUTING AGENT

route_result = routing_agent.plan_route(final_decisions)

# 5. FINAL ROUTE

print("\n FINAL COLLECTION ROUTE \n")

print(route_result["message"])

for index, stop in enumerate(route_result["route"], start=1):

    print(f"\nSTOP {index}")
    print(f"Bin: {stop['bin_id']}")
    print(f"Priority: {stop['priority']}")
    print(f"Action: {stop['action']}")
    print(f"Distance: {stop['distance_from_previous_km']} km")


print(f"Total route distance: {route_result['total_distance_km']} km")
