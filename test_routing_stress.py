
import random
import time
import statistics

from routing_agent.routing_module import RoutingAgent


# ==================================================
# CONFIGURATION
# ==================================================

DEPOT_LAT = 36.7500
DEPOT_LON = 3.0500

NUMBER_OF_SCENARIOS = 50
MIN_BINS = 5
MAX_BINS = 30

BENCHMARK_REPETITIONS = 30

PRIORITY_ORDER = [
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW"
]

PRIORITY_VALUE = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1
}

agent = RoutingAgent(DEPOT_LAT, DEPOT_LON)


# ==================================================
# 1. GENERATE RANDOM SCENARIOS
# ==================================================

def generate_scenario(seed):

    random.seed(seed)

    number_of_bins = random.randint(
        MIN_BINS,
        MAX_BINS
    )

    bins = []

    for i in range(number_of_bins):

        priority = random.choices(
            PRIORITY_ORDER,
            weights=[10, 30, 35, 25],
            k=1
        )[0]

        # Simulate bins located around the depot
        latitude = DEPOT_LAT + random.uniform(-0.03, 0.03)
        longitude = DEPOT_LON + random.uniform(-0.03, 0.03)

        bins.append({
            "bin_id": f"BIN_{i+1:03d}",
            "priority": priority,
            "action": "COLLECT",
            "collection_required": True,
            "latitude": latitude,
            "longitude": longitude
        })

    # Add a few bins that do not require collection
    number_of_non_collection_bins = random.randint(0, 3)

    for i in range(number_of_non_collection_bins):

        bins.append({
            "bin_id": f"MONITOR_{i+1:03d}",
            "priority": "LOW",
            "action": "MONITOR",
            "collection_required": False,
            "latitude": DEPOT_LAT + random.uniform(-0.03, 0.03),
            "longitude": DEPOT_LON + random.uniform(-0.03, 0.03)
        })

    return bins


# ==================================================
# 2. BASELINE: ORIGINAL PRIORITY NEAREST NEIGHBOR
# ==================================================

def baseline_route(decisions):

    remaining = [
        b for b in decisions
        if b.get("collection_required", False)
    ]

    route = []

    total_distance = 0

    current_lat = DEPOT_LAT
    current_lon = DEPOT_LON

    while remaining:

        highest_priority = max(
            PRIORITY_VALUE.get(
                b.get("priority", "LOW"), 1
            )
            for b in remaining
        )

        candidates = [
            b for b in remaining
            if PRIORITY_VALUE.get(
                b.get("priority", "LOW"), 1
            ) == highest_priority
        ]

        nearest = min(
            candidates,
            key=lambda b: agent.calculate_distance(
                current_lat,
                current_lon,
                b["latitude"],
                b["longitude"]
            )
        )

        distance = agent.calculate_distance(
            current_lat,
            current_lon,
            nearest["latitude"],
            nearest["longitude"]
        )

        total_distance += distance

        route.append(nearest["bin_id"])

        current_lat = nearest["latitude"]
        current_lon = nearest["longitude"]

        remaining.remove(nearest)

    return_distance = agent.calculate_distance(
        current_lat,
        current_lon,
        DEPOT_LAT,
        DEPOT_LON
    )

    total_distance += return_distance

    return {
        "route": route,
        "total_distance_km": total_distance
    }


# ==================================================
# 3. OPTIMIZED ROUTING AGENT
# ==================================================


def optimized_route(decisions):

    result = agent.plan_route(decisions)

    # Recalculate the distance using full precision.
    # Do not use the rounded total_distance_km
    # returned for display purposes.
    exact_distance = agent.calculate_route_distance(
        result["route"]
    )

    return {
        "route": [
            stop["bin_id"]
            for stop in result["route"]
        ],
        "total_distance_km": exact_distance,
        "details": result
    }



# ==================================================
# 4. VALIDATE ROUTE
# ==================================================

def validate_route(decisions, result):

    required_bins = [
        b["bin_id"]
        for b in decisions
        if b.get("collection_required", False)
    ]

    visited_bins = result["route"]

    # All collection bins must be visited
    all_visited = set(required_bins) == set(visited_bins)

    # No duplicate visits
    no_duplicates = len(visited_bins) == len(set(visited_bins))

    # Priority order must never decrease
    priorities = [
        next(
            b["priority"]
            for b in decisions
            if b["bin_id"] == bin_id
        )
        for bin_id in visited_bins
    ]

    priority_values = [
        PRIORITY_VALUE[p]
        for p in priorities
    ]

    priority_respected = all(
        priority_values[i] >= priority_values[i + 1]
        for i in range(len(priority_values) - 1)
    )

    return (
        all_visited
        and no_duplicates
        and priority_respected
    )


# ==================================================
# 5. BENCHMARK EXECUTION TIME
# ==================================================

def measure_execution_time(function, decisions):

    times = []

    for _ in range(BENCHMARK_REPETITIONS):

        start = time.perf_counter()

        function(decisions)

        end = time.perf_counter()

        times.append((end - start) * 1000)

    return statistics.mean(times)


# ==================================================
# 6. RUN STRESS TEST
# ==================================================

results = []

valid_routes = 0

optimized_better = 0
optimized_equal = 0
optimized_worse = 0

total_bins_collected = 0

total_distance_saved = 0

total_baseline_distance = 0
total_optimized_distance = 0

baseline_times = []
optimized_times = []

print("\n========== ROUTING STRESS TEST ==========\n")


for scenario_number in range(1, NUMBER_OF_SCENARIOS + 1):

    scenario = generate_scenario(scenario_number)

    baseline = baseline_route(scenario)

    optimized = optimized_route(scenario)

    baseline_distance = baseline["total_distance_km"]

    optimized_distance = optimized["total_distance_km"]

    distance_saved = (
        baseline_distance - optimized_distance
    )

    # Percentage improvement
    percentage_saved = (
        distance_saved / baseline_distance * 100
        if baseline_distance > 0
        else 0
    )

    # Route validation
    is_valid = validate_route(
        scenario,
        optimized
    )

    if is_valid:
        valid_routes += 1

    # Compare results
    if distance_saved > 0.000001:

        optimized_better += 1

    elif distance_saved < -0.000001:

        optimized_worse += 1

    else:

        optimized_equal += 1

    total_bins_collected += len(
        baseline["route"]
    )

    total_distance_saved += distance_saved

    total_baseline_distance += baseline_distance

    total_optimized_distance += optimized_distance

    # Measure execution time
    baseline_time = measure_execution_time(
        baseline_route,
        scenario
    )

    optimized_time = measure_execution_time(
        optimized_route,
        scenario
    )

    baseline_times.append(baseline_time)

    optimized_times.append(optimized_time)

    results.append({
        "scenario": scenario_number,
        "bins": len(baseline["route"]),
        "baseline_km": baseline_distance,
        "optimized_km": optimized_distance,
        "saved_km": distance_saved,
        "saved_percent": percentage_saved,
        "valid": is_valid
    })

    print(
        f"Scenario {scenario_number:02d} | "
        f"Bins: {len(baseline['route']):02d} | "
        f"Baseline: {baseline_distance:.3f} km | "
        f"Optimized: {optimized_distance:.3f} km | "
        f"Saved: {distance_saved:.3f} km "
        f"({percentage_saved:.2f}%) | "
        f"Valid: {is_valid}"
    )


# ==================================================
# 7. FINAL STATISTICS
# ==================================================

average_baseline_distance = (
    total_baseline_distance / NUMBER_OF_SCENARIOS
)

average_optimized_distance = (
    total_optimized_distance / NUMBER_OF_SCENARIOS
)

overall_improvement = (
    total_distance_saved / total_baseline_distance * 100
    if total_baseline_distance > 0
    else 0
)

average_saving_per_scenario = (
    total_distance_saved / NUMBER_OF_SCENARIOS
)


# ==================================================
# 8. DISPLAY SUMMARY
# ==================================================

print("\n")
print("============================================")
print("         FINAL STRESS TEST SUMMARY")
print("============================================")

print("Scenarios tested:", NUMBER_OF_SCENARIOS)

print("Total bins collected:", total_bins_collected)

print("Valid optimized routes:", valid_routes)

print("Invalid optimized routes:",
      NUMBER_OF_SCENARIOS - valid_routes)

print("\n--- DISTANCE PERFORMANCE ---")

print("Average baseline distance:",
      round(average_baseline_distance, 3), "km")

print("Average optimized distance:",
      round(average_optimized_distance, 3), "km")

print("Total distance saved:",
      round(total_distance_saved, 3), "km")

print("Average saving per scenario:",
      round(average_saving_per_scenario, 3), "km")

print("Overall distance improvement:",
      round(overall_improvement, 2), "%")


print("\n--- SCENARIO COMPARISON ---")

print("Optimized better:", optimized_better)

print("Equal distance:", optimized_equal)

print("Optimized worse:", optimized_worse)


print("\n--- EXECUTION TIME ---")

print("Baseline average:",
      round(statistics.mean(baseline_times), 6), "ms")

print("Optimized average:",
      round(statistics.mean(optimized_times), 6), "ms")

print("Baseline median:",
      round(statistics.median(baseline_times), 6), "ms")

print("Optimized median:",
      round(statistics.median(optimized_times), 6), "ms")


print("\n============================================")
