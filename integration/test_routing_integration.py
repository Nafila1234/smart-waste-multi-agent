
from routing_agent.routing_module import RoutingAgent


# 1. Example decisions from Agent 3 (LLM)
decisions = [
    {
        "bin_id": "BIN_001",
        "collection_required": True,
        "priority": "HIGH",
        "action": "COLLECT",
        "latitude": 36.7665,
        "longitude": 3.4772
    },
    {
        "bin_id": "BIN_002",
        "collection_required": True,
        "priority": "CRITICAL",
        "action": "IMMEDIATE_COLLECTION",
        "latitude": 36.7680,
        "longitude": 3.4850
    },
    {
        "bin_id": "BIN_003",
        "collection_required": False,
        "priority": "LOW",
        "action": "MONITOR",
        "latitude": 36.7600,
        "longitude": 3.4700
    },
    {
        "bin_id": "BIN_004",
        "collection_required": True,
        "priority": "MEDIUM",
        "action": "SCHEDULE_COLLECTION",
        "latitude": 36.7550,
        "longitude": 3.4800
    }
]


# 2. Initialize Routing Module
# Example depot coordinates
routing_module = RoutingAgent(
    depot_latitude=36.7660,
    depot_longitude=3.4720
)


# 3. Generate and optimize the route
route_result = routing_module.plan_route(decisions)


# 4. Display results
print("\n========== ROUTING MODULE RESULTS ==========")

print("\nOptimized collection route:")

for index, stop in enumerate(route_result["route"], start=1):
    print(f"\nStop {index}")
    print(f"Bin ID: {stop['bin_id']}")
    print(f"Priority: {stop['priority']}")
    print(f"Action: {stop['action']}")
    print(
        f"Distance from previous stop: "
        f"{stop['distance_from_previous_km']} km"
    )

print("\n========== ROUTE SUMMARY ==========")
print(f"Total distance: {route_result['total_distance_km']} km")
print(f"Baseline distance: {route_result['baseline_distance_km']} km")
print(f"Return distance: {route_result['return_distance_km']} km")
print(f"Optimization iterations: {route_result['optimization_iterations']}")
print(route_result["message"])
