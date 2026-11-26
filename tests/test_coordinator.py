"""
Test of the Coordinator on simulated bins (no image needed).
Run from the repository root:  python tests/test_coordinator.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integration.coordinator import Coordinator


def make_bin(bin_id, fill, prev_fill, waste, confidence, lat, lon,
             minutes=30, start=60.0, overflow=False):
    return {
        "bin_id": bin_id,
        "current": {
            "fill_level": fill, "weight": 25.0, "energy_used": 12.0,
            "time_diff": 10, "minutes": minutes, "niveau_initial": start,
        },
        "previous": {"fill_level": prev_fill, "weight": 23.0},
        "visual": {"waste_type": waste, "confidence": confidence,
                   "hazard_detected": False},
        "sensor": {"overflow_flag": overflow},
        "latitude": lat,
        "longitude": lon,
    }


bins_data = [
    make_bin("BIN_001", 78.0, 75.0, "plastic", 0.62, 36.7510, 3.0520),
    make_bin("BIN_002", 92.0, 89.0, "organic", 0.97, 36.7540, 3.0550, start=70.0),
    make_bin("BIN_003", 30.0, 28.0, "paper", 0.95, 36.7520, 3.0510,
             minutes=0, start=20.0),
    make_bin("BIN_004", 66.0, 64.0, "battery", 0.95, 36.7550, 3.0540, start=40.0),
]

coordinator = Coordinator(depot_latitude=36.7500, depot_longitude=3.0500)
result = coordinator.run(bins_data)

print("\n=== DECISIONS ===")
for d in result["decisions"]:
    print(f"{d['bin_id']}: {d['waste_type']:8s} priority={d['priority']:8s} "
          f"collect={d['collection_required']} review={d['needs_review']}")

print("\n=== ROUTE ===")
for i, stop in enumerate(result["route"]["route"], start=1):
    print(f"{i}. {stop['bin_id']} ({stop['priority']}) "
          f"{stop['distance_from_previous_km']} km")
print("Total distance:", result["route"]["total_distance_km"], "km")
print("Needs review:", result["needs_review"])
print("Errors:", result["errors"])

# --- Checks that do not depend on the trained models ---
by_id = {d["bin_id"]: d for d in result["decisions"]}

assert result["errors"] == [], "no bin should fail"
assert len(result["decisions"]) == 4

# a battery must be treated as hazardous and collected first
assert by_id["BIN_004"]["hazard_detected"] is True
assert by_id["BIN_004"]["priority"] == "CRITICAL"
assert result["route"]["route"][0]["bin_id"] == "BIN_004"

# only the low-confidence prediction is flagged for review
assert result["needs_review"] == ["BIN_001"]

print("\nAll checks passed.")
