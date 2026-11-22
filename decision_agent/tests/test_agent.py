from decision_agent.decision_agent import DecisionAgent
agent = DecisionAgent()

test_cases = [
    {
        "name": "Normal bin",
        "prediction": {
            "bin_id": "BIN_002",
            "current_fill": 40,
            "predicted_fill": 55
        },
        "visual": {
            "waste_type": "paper",
            "hazard_detected": False
        },
        "sensor": {"overflow_flag": False}
    },
    {
        "name": "Full bin",
        "prediction": {
            "bin_id": "BIN_003",
            "current_fill": 90,
            "predicted_fill": 95
        },
        "visual": {
            "waste_type": "plastic",
            "hazard_detected": False
        },
        "sensor": {"overflow_flag": False}
    },
    {
        "name": "Hazardous waste",
        "prediction": {
            "bin_id": "BIN_004",
            "current_fill": 30,
            "predicted_fill": 40
        },
        "visual": {
            "waste_type": "hazardous",
            "hazard_detected": True
        },
        "sensor": {"overflow_flag": False}
    },
    {
        "name": "Overflow",
        "prediction": {
            "bin_id": "BIN_005",
            "current_fill": 70,
            "predicted_fill": 80
        },
        "visual": {
            "waste_type": "organic",
            "hazard_detected": False
        },
        "sensor": {"overflow_flag": True}
    }
]

for test in test_cases:
    result = agent.evaluate_bin(
        test["prediction"],
        test["visual"],
        test["sensor"]
    )

    print("\n---", test["name"], "---")
    print("Bin:", result["bin_id"])
    print("Collection:", result["collection_required"])
    print("Priority:", result["priority"])
    print("Action:", result["action"])
    print("Reason:", result["reason"])