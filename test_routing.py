from routing_agent.routing_agent import RoutingAgent


# Simulated depot location
routing_agent = RoutingAgent(
    depot_latitude=36.7500,
    depot_longitude=3.0500
)

# Simulated decisions received from Decision Agent
decisions = [
    {
        "bin_id": "BIN_001",
        "collection_required": True,
        "priority": "HIGH",
        "action": "SCHEDULE_COLLECTION",
        "latitude": 36.7510,
        "longitude": 3.0520
    },
    {
        "bin_id": "BIN_002",
        "collection_required": True,
        "priority": "CRITICAL",
        "action": "IMMEDIATE_COLLECTION",
        "latitude": 36.7540,
        "longitude": 3.0550
    },
    {
        "bin_id": "BIN_003",
        "collection_required": True,
        "priority": "HIGH",
        "action": "COLLECT",
        "latitude": 36.7520,
        "longitude": 3.0510
    },
    {
        "bin_id": "BIN_004",
        "collection_required": False,
        "priority": "LOW",
        "action": "MONITOR",
        "latitude": 36.7600,
        "longitude": 3.0600
    }
]


# Generate route
result = routing_agent.plan_route(decisions)


# Display results
print("\n========== ROUTING AGENT RESULTS ==========\n")

print(result["message"])

print("\nOptimal Collection Order:")

for i, stop in enumerate(result["route"], start=1):

    print(f"\nStop {i}: {stop['bin_id']}")
    print(f"Priority: {stop['priority']}")
    print(f"Action: {stop['action']}")
    print(f"Distance from previous stop: "
          f"{stop['distance_from_previous_km']} km")

print(f"Return to depot: {result['return_distance_km']} km")
print(f"Total round-trip distance: {result['total_distance_km']} km")