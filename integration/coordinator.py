"""
Coordinator of the Smart Waste Collection multi-agent system.

For every bin :
    photo   -> VisualAdapter   -> waste type, confidence, hazard flag
    sensors -> PredictionAgent -> fill forecast, tree / regression signals
    both    -> DecisionAgent   -> priority and action
Then all decisions -> RoutingAgent -> ordered collection route.

Place this file in the `integration/` folder of the repository.
Run scripts from the repository root, for example:
    python tests/test_coordinator.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DEFAULT_WEIGHTS = (
    BASE_DIR / "prediction_agent" / "models" / "vision_agent_resnet18_dedup.pkl"
)

# A prediction below this confidence is flagged for human review.
REVIEW_THRESHOLD = 0.80

# Classes of the vision agent that count as hazardous waste.
HAZARDOUS_CLASSES = {"battery"}

UNKNOWN_VISUAL = {
    "waste_type": "unknown",
    "confidence": None,
    "hazard_detected": False,
}


class Coordinator:
    """Chains the four agents and returns decisions plus the route."""

    def __init__(
        self,
        depot_latitude=36.7500,
        depot_longitude=3.0500,
        weights_path=DEFAULT_WEIGHTS,
        review_threshold=REVIEW_THRESHOLD,
        prediction_agent=None,
        decision_agent=None,
        routing_agent=None,
        visual_adapter=None,
    ):
        self.review_threshold = review_threshold

        # Agents can be injected (useful for tests); otherwise the real ones are built.
        if prediction_agent is None:
            from prediction_agent.prediction_agent import PredictionAgent
            prediction_agent = PredictionAgent()

        if decision_agent is None:
            from decision_agent.decision_agent import DecisionAgent
            decision_agent = DecisionAgent()

        if routing_agent is None:
            from routing_agent.routing_module import RoutingAgent
            routing_agent = RoutingAgent(depot_latitude, depot_longitude)

        # The vision agent is optional: without weights, bins must bring
        # their own "visual" dictionary (or are treated as unknown waste).
        if visual_adapter is None and weights_path and Path(weights_path).exists():
            try:
                from integration.visual_adapter import VisualAdapter
            except ImportError:
                from visual_adapter import VisualAdapter
            visual_adapter = VisualAdapter(str(weights_path))

        self.prediction_agent = prediction_agent
        self.decision_agent = decision_agent
        self.routing_agent = routing_agent
        self.visual_adapter = visual_adapter

    # ------------------------------------------------------------------
    def analyze_visual(self, record):
        """Return the visual information of one bin as a dictionary."""
        if record.get("image") is not None:
            if self.visual_adapter is None:
                raise RuntimeError(
                    "An image was given but the vision weights were not found."
                )
            visual = dict(self.visual_adapter.analyze(record["image"]))
        elif record.get("visual"):
            visual = dict(record["visual"])
        else:
            visual = dict(UNKNOWN_VISUAL)

        waste_type = str(visual.get("waste_type", "unknown")).lower()
        confidence = visual.get("confidence")
        visual["waste_type"] = waste_type

        # Batteries are hazardous: the CNN has no separate hazard output,
        # so the hazard flag is derived from the predicted class. It is kept
        # even when confidence is low, because missing a battery is worse
        # than one extra collection.
        visual.setdefault("hazard_detected", False)
        if waste_type in HAZARDOUS_CLASSES:
            visual["hazard_detected"] = True

        visual["needs_review"] = (
            confidence is not None and confidence < self.review_threshold
        )
        return visual

    # ------------------------------------------------------------------
    def process_bin(self, record):
        """Run visual, prediction and decision agents for one bin.

        Expected record keys: bin_id, current, previous, latitude, longitude,
        and optionally sensor, image, visual.
        """
        visual = self.analyze_visual(record)

        current = {"bin_id": record["bin_id"], **record["current"]}
        prediction = self.prediction_agent.predict_bin(current, record["previous"])

        decision = self.decision_agent.evaluate_bin(
            prediction, visual, record.get("sensor", {})
        )

        # Extra fields needed by the routing agent and the interface
        decision["latitude"] = record["latitude"]
        decision["longitude"] = record["longitude"]
        decision["visual_confidence"] = visual.get("confidence")
        decision["needs_review"] = visual["needs_review"]
        decision["remaining_time_minutes"] = prediction.get("remaining_time_minutes")
        return decision

    # ------------------------------------------------------------------
    def run(self, bins_data):
        """Process all bins, then plan the route.

        A bin that fails is reported in `errors` and does not stop the others.
        """
        decisions, errors = [], []

        for record in bins_data:
            try:
                decisions.append(self.process_bin(record))
            except Exception as exc:
                errors.append({"bin_id": record.get("bin_id"), "error": str(exc)})

        route = self.routing_agent.plan_route(decisions)
        review = [d["bin_id"] for d in decisions if d.get("needs_review")]

        return {
            "decisions": decisions,
            "route": route,
            "needs_review": review,
            "errors": errors,
        }
