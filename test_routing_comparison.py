
import time
import itertools

from routing_agent.routing_agent import RoutingAgent


# ==========================================
# 1. DEPOT
# ==========================================

DEPOT_LAT = 36.7500
DEPOT_LON = 3.0500

routing_agent = RoutingAgent(DEPOT_LAT, DEPOT_LON)


# ==========================================
# 2. SIMULATED DATASET (10 BINS)
# ==========================================

bins = [
    # CRITICAL
    {
        "bin_id": "BIN_C1",
        "priority": "CRITICAL",
        "action": "IMMEDIATE_COLLECTION",
        "collection_required": True,
        "latitude": 36.742394,
        "longitude": 3.055079
    },
    {
        "bin_id": "BIN_C2",
        "priority": "CRITICAL",
        "action": "IMMEDIATE_COLLECTION",
        "collection_required": True,
        "latitude": 36.759276,
        "longitude": 3.064186
    },

    # HIGH
    {
        "bin_id": "BIN_H1",
        "priority": "HIGH",
        "action": "COLLECT",
        "collection_required": True,
        "latitude": 36.765202,
        "longitude": 3.033469
    },
    {
        "bin_id": "BIN_H2",
        "priority": "HIGH",
        "action": "COLLECT",
        "collection_required": True,
        "latitude": 36.754234,
        "longitude": 3.056868
    },
    {
        "bin_id": "BIN_H3",
        "priority": "HIGH",
        "action": "COLLECT",
        "collection_required": True,
        "latitude": 36.750238,
        "longitude": 3.037112
    },

    # MEDIUM
    {
        "bin_id": "BIN_M1",
        "priority": "MEDIUM",
        "action": "SCHEDULE_COLLECTION",
        "collection_required": True,
        "latitude": 36.748944,
        "longitude": 3.033574
    },
    {
        "bin_id": "BIN_M2",
        "priority": "MEDIUM",
        "action": "SCHEDULE_COLLECTION",
        "collection_required": True,
        "latitude": 36.767384,
        "longitude": 3.064619
    },
    {
        "bin_id": "BIN_M3",
        "priority": "MEDIUM",
        "action": "SCHEDULE_COLLECTION",
        "collection_required": True,
        "latitude": 36.751906,
        "longitude": 3.042010
    },

    # LOW
    {
        "bin_id": "BIN_L1",
        "priority": "LOW",
        "action": "MONITOR",
        "collection_required": True,
        "latitude": 36.766355,
        "longitude": 3.052895
    },
    {
        "bin_id": "BIN_L2",
        "priority": "LOW",
        "action": "MONITOR",
        "collection_required": True,
        "latitude": 36.765293,
        "longitude": 3.063922
    }
]


# ==========================================
# 3. CURRENT ALGORITHM
# ==========================================

def current_algorithm():

    return routing_agent.plan_route(bins)


# ==========================================
# 4. OPTIMIZED ALGORITHM
# ==========================================

def optimized_algorithm():

    priority_order = [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW"
    ]

    remaining_bins = bins.copy()

    route = []
    total_distance = 0

    current_lat = DEPOT_LAT
    current_lon = DEPOT_LON

    for priority in priority_order:

        priority_bins = [
            b for b in remaining_bins
            if b["priority"] == priority
        ]

        if not priority_bins:
            continue

        # Test every possible order within this priority group
        possible_routes = itertools.permutations(priority_bins)

        best_route = None
        best_distance = float("inf")

        for candidate in possible_routes:

            candidate_distance = 0

            temp_lat = current_lat
            temp_lon = current_lon

            for bin_data in candidate:

                distance = routing_agent.calculate_distance(
                    temp_lat,
                    temp_lon,
                    bin_data["latitude"],
                    bin_data["longitude"]
                )

                candidate_distance += distance

                temp_lat = bin_data["latitude"]
                temp_lon = bin_data["longitude"]

            if candidate_distance < best_distance:

                best_distance = candidate_distance
                best_route = candidate

        # Execute the best order found for this priority
        for bin_data in best_route:

            distance = routing_agent.calculate_distance(
                current_lat,
                current_lon,
                bin_data["latitude"],
                bin_data["longitude"]
            )

            total_distance += distance

            route.append({
                "bin_id": bin_data["bin_id"],
                "priority": bin_data["priority"],
                "action": bin_data["action"],
                "latitude": bin_data["latitude"],
                "longitude": bin_data["longitude"],
                "distance_from_previous_km": round(distance, 3)
            })

            current_lat = bin_data["latitude"]
            current_lon = bin_data["longitude"]

            remaining_bins.remove(bin_data)

    # Return to depot
    return_distance = routing_agent.calculate_distance(
        current_lat,
        current_lon,
        DEPOT_LAT,
        DEPOT_LON
    )

    total_distance += return_distance

    return {
        "route": route,
        "return_distance_km": round(return_distance, 3),
        "total_distance_km": round(total_distance, 3),
        "message": "Optimized route generated successfully."
    }


# ==========================================
# 5. BENCHMARK EXECUTION
# ==========================================

def benchmark(function, repetitions=1000):

    start = time.perf_counter()

    for _ in range(repetitions):
        function()

    end = time.perf_counter()

    average_ms = ((end - start) / repetitions) * 1000

    return average_ms


# ==========================================
# 6. RUN COMPARISON
# ==========================================

current_result = current_algorithm()
optimized_result = optimized_algorithm()


print("\n========== ROUTE COMPARISON ==========")


print("\n--- Current Algorithm ---")

for stop in current_result["route"]:
    print(stop["bin_id"], stop["priority"])

print("Total distance:",
      current_result["total_distance_km"], "km")


print("\n--- Optimized Algorithm ---")

for stop in optimized_result["route"]:
    print(stop["bin_id"], stop["priority"])

print("Total distance:",
      optimized_result["total_distance_km"], "km")


# ==========================================
# 7. DISTANCE PERFORMANCE
# ==========================================

current_distance = current_result["total_distance_km"]

optimized_distance = optimized_result["total_distance_km"]

distance_saved = current_distance - optimized_distance

if current_distance > 0:

    percentage_saved = (
        distance_saved / current_distance
    ) * 100

else:
    percentage_saved = 0


# ==========================================
# 8. EXECUTION TIME
# ==========================================

REPETITIONS = 1000

current_time = benchmark(
    current_algorithm,
    REPETITIONS
)

optimized_time = benchmark(
    optimized_algorithm,
    REPETITIONS
)


# ==========================================
# 9. FINAL RESULTS
# ==========================================

print("\n========== PERFORMANCE ==========")

print("Distance saved:",
      round(distance_saved, 3), "km")

print("Percentage saved:",
      round(percentage_saved, 2), "%")

print("Bins collected:",
      len(current_result["route"]))

print("Current average execution:",
      round(current_time, 6), "ms")

print("Optimized average execution:",
      round(optimized_time, 6), "ms")

if optimized_time > 0:

    speed_ratio = current_time / optimized_time

    print("Speed ratio:",
          round(speed_ratio, 2), "x")

print("\n================================")
