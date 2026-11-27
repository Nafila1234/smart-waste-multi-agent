
import math


class RoutingAgent:

    PRIORITY_ORDER = {
        "CRITICAL": 4,
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1
    }

    PRIORITY_LEVELS = [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW"
    ]

    MAX_ITERATIONS = 100
    EPSILON = 1e-9

    def __init__(self, depot_latitude, depot_longitude):

        self.depot = {
            "latitude": depot_latitude,
            "longitude": depot_longitude
        }

    # 1. Calculate distance using Haversine

    def calculate_distance(self, lat1, lon1, lat2, lon2):

        earth_radius = 6371

        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)

        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1)
            * math.cos(lat2)
            * math.sin(dlon / 2) ** 2
        )

        c = 2 * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a)
        )

        return earth_radius * c

    # 2. Normalize priority

    def get_priority(self, bin_data):

        priority = bin_data.get("priority", "LOW")

        if priority not in self.PRIORITY_ORDER:
            return "LOW"

        return priority
    # 3. Generate initial route
    #    Priority + Nearest Neighbor

    def generate_initial_route(self, bins_to_collect):

        priority_groups = {}

        for bin_data in bins_to_collect:

            priority = self.get_priority(bin_data)

            priority_groups.setdefault(
                priority, []
            ).append(bin_data)

        route = []

        current_lat = self.depot["latitude"]
        current_lon = self.depot["longitude"]

        for priority in self.PRIORITY_LEVELS:

            group = priority_groups.get(priority, [])

            remaining = group.copy()

            while remaining:

                nearest_bin = min(
                    remaining,
                    key=lambda bin_data: self.calculate_distance(
                        current_lat,
                        current_lon,
                        bin_data["latitude"],
                        bin_data["longitude"]
                    )
                )

                route.append(nearest_bin)

                current_lat = nearest_bin["latitude"]
                current_lon = nearest_bin["longitude"]

                remaining.remove(nearest_bin)

        return route

    # 4. Calculate total route distance
    #    INCLUDING return to depot

    def calculate_route_distance(self, route):

        if not route:
            return 0

        total_distance = 0

        current_lat = self.depot["latitude"]
        current_lon = self.depot["longitude"]

        for bin_data in route:

            distance = self.calculate_distance(
                current_lat,
                current_lon,
                bin_data["latitude"],
                bin_data["longitude"]
            )

            total_distance += distance

            current_lat = bin_data["latitude"]
            current_lon = bin_data["longitude"]

        # Return to depot
        total_distance += self.calculate_distance(
            current_lat,
            current_lon,
            self.depot["latitude"],
            self.depot["longitude"]
        )

        return total_distance

    # 5. Priority-Constrained 2-opt

    def optimize_route(self, route):

        route = route.copy()

        current_distance = self.calculate_route_distance(route)

        iterations = 0

        while iterations < self.MAX_ITERATIONS:

            best_route = None
            best_distance = current_distance

            # Search for the best improving reversal
            for i in range(len(route)):

                priority_i = self.get_priority(route[i])

                for j in range(i + 1, len(route)):

                    priority_j = self.get_priority(route[j])

                    # Reversals must stay inside
                    # the same priority group
                    if priority_i != priority_j:
                        break

                    candidate = route.copy()

                    candidate[i:j + 1] = reversed(
                        candidate[i:j + 1]
                    )

                    candidate_distance = (
                        self.calculate_route_distance(candidate)
                    )

                    # Accept only strict improvements
                    if (
                        candidate_distance
                        < best_distance - self.EPSILON
                    ):

                        best_distance = candidate_distance
                        best_route = candidate

            # No improvement found
            if best_route is None:
                break

            # Apply best improvement
            route = best_route
            current_distance = best_distance

            iterations += 1

        return route, iterations

    # 6. Main route planning function

    def plan_route(self, decisions):

        bins_to_collect = [
            bin_data
            for bin_data in decisions
            if bin_data.get("collection_required", False)
        ]

        # No bins require collection
        if not bins_to_collect:

            return {
                "route": [],
                "return_distance_km": 0,
                "total_distance_km": 0,
                "baseline_distance_km": 0,
                "optimization_iterations": 0,
                "message": "No bins require collection."
            }

        # Validate coordinates
        for bin_data in bins_to_collect:

            if (
                "latitude" not in bin_data
                or "longitude" not in bin_data
            ):

                raise ValueError(
                    f"Missing coordinates for "
                    f"{bin_data.get('bin_id')}"
                )

        # Generate initial solution
        initial_route = self.generate_initial_route(
            bins_to_collect
        )

        baseline_distance = self.calculate_route_distance(
            initial_route
        )

        # Optimize using priority-constrained 2-opt
        optimized_route, iterations = self.optimize_route(
            initial_route
        )

        total_distance = self.calculate_route_distance(
            optimized_route
        )

        # Build final route details
        route_details = []

        current_lat = self.depot["latitude"]
        current_lon = self.depot["longitude"]

        for bin_data in optimized_route:

            distance = self.calculate_distance(
                current_lat,
                current_lon,
                bin_data["latitude"],
                bin_data["longitude"]
            )

            route_details.append({
                "bin_id": bin_data["bin_id"],
                "priority": self.get_priority(bin_data),
                "action": bin_data.get("action", "COLLECT"),
                "latitude": bin_data["latitude"],
                "longitude": bin_data["longitude"],
                "distance_from_previous_km": round(
                    distance, 3
                )
            })

            current_lat = bin_data["latitude"]
            current_lon = bin_data["longitude"]

        # Distance from final bin back to depot
        return_distance = self.calculate_distance(
            current_lat,
            current_lon,
            self.depot["latitude"],
            self.depot["longitude"]
        )

        return {
            "route": route_details,
            "return_distance_km": round(
                return_distance, 3
            ),
            "total_distance_km": round(
                total_distance, 3
            ),
            "baseline_distance_km": round(
                baseline_distance, 3
            ),
            "optimization_iterations": iterations,
            "message": (
                "Priority-constrained 2-opt "
                "optimization completed successfully."
            )
        }
