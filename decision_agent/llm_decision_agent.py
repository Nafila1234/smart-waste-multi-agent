
import os
import json

from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

from decision_agent.config import FILL_THRESHOLD, PRIORITIES


# Load API key from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "openai/gpt-oss-120b"

VALID_PRIORITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

VALID_ACTIONS = [
    "MONITOR",
    "SCHEDULE_COLLECTION",
    "COLLECT",
    "IMMEDIATE_COLLECTION"
]


class LLMDecisionAgent:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")

        self.client = Groq(api_key=api_key)

    def _safety_decision(self, prediction, visual, sensor):
        """Apply deterministic safety rules."""

        hazard = sensor.get("hazard_detected", False)
        overflow = sensor.get("overflow_flag", False)

        if hazard or overflow:
            reason = (
                "Hazardous waste detected by sensor."
                if hazard
                else "Bin overflow detected."
            )

            return {
                "bin_id": prediction.get("bin_id"),
                "current_fill": prediction.get("current_fill"),
                "predicted_fill": prediction.get(
                    "predicted_fill_10min"
                ),
                "waste_type": visual.get("waste_type", "Unknown"),
                "visual_confidence": visual.get("confidence", 0),
                "hazard_detected": hazard,
                "overflow_detected": overflow,
                "collection_required": True,
                "priority": "CRITICAL",
                "priority_value": PRIORITIES["CRITICAL"],
                "action": "IMMEDIATE_COLLECTION",
                "reason": reason,
                "decision_source": "SAFETY_RULE"
            }

        return None

    def evaluate_bin(self, prediction, visual, sensor):

        # 1. Safety checks always have priority over the LLM
        safety_result = self._safety_decision(
            prediction, visual, sensor
        )

        if safety_result:
            return safety_result

        # 2. Prepare the information for the LLM
        context = {
            "prediction": prediction,
            "visual": visual,
            "sensor": sensor
        }

        system_prompt = """
You are an intelligent urban waste collection decision agent.

Your task is to analyze:
- Current bin fill level
- Predicted future fill level
- Decision Tree prediction
- Linear Regression prediction
- Waste category and visual confidence
- Sensor information

Make a practical waste collection decision.

Rules:
1. If current fill is at or above 85%, collection is required.
2. If both prediction models indicate collection, prioritize collection.
3. If only one model indicates collection, consider scheduling collection.
4. If collection is not necessary, choose MONITOR and LOW priority.
5. Use:
   - LOW: monitoring
   - MEDIUM: scheduled collection
   - HIGH: urgent collection
   - CRITICAL: immediate collection
6. Do not invent sensor information.
7. Explain your decision briefly.

Return ONLY a valid JSON object with these fields:
{
  "collection_required": true,
  "priority": "HIGH",
  "action": "SCHEDULE_COLLECTION",
  "reason": "Explanation"
}

Allowed priorities: LOW, MEDIUM, HIGH, CRITICAL.

Allowed actions:
MONITOR, SCHEDULE_COLLECTION, COLLECT, IMMEDIATE_COLLECTION.
"""

        # 3. Ask Groq to make the decision
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": json.dumps(context)
                }
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=2048
        )

        # 4. Parse the JSON response
        raw_result = response.choices[0].message.content

        if not raw_result:
            raise ValueError("Groq returned an empty response.")

        result = json.loads(raw_result)

        # 5. Validate the LLM response
        required_fields = [
            "collection_required",
            "priority",
            "action",
            "reason"
        ]

        for field in required_fields:
            if field not in result:
                raise ValueError(
                    f"Missing LLM response field: {field}"
                )

        if not isinstance(result["collection_required"], bool):
            raise ValueError("collection_required must be boolean.")

        if result["priority"] not in VALID_PRIORITIES:
            raise ValueError("Invalid priority returned by LLM.")

        if result["action"] not in VALID_ACTIONS:
            raise ValueError("Invalid action returned by LLM.")

        if not isinstance(result["reason"], str) or not result["reason"].strip():
            raise ValueError("Invalid decision reason.")

        # 6. Validate consistency between the decision fields
        if result["collection_required"]:
            if result["priority"] == "LOW":
                raise ValueError(
                    "Collection cannot have LOW priority."
                )

            if result["action"] == "MONITOR":
                raise ValueError(
                    "Collection cannot have MONITOR action."
                )

        else:
            if result["priority"] != "LOW":
                raise ValueError(
                    "A non-collection decision must have LOW priority."
                )

            if result["action"] != "MONITOR":
                raise ValueError(
                    "A non-collection decision must use MONITOR."
                )

        # 7. Enforce the collection threshold independently
        current_fill = float(prediction.get("current_fill", 0))

        if current_fill >= FILL_THRESHOLD:
            return {
                "bin_id": prediction.get("bin_id"),
                "current_fill": current_fill,
                "predicted_fill": prediction.get("predicted_fill_10min"),
                "waste_type": visual.get("waste_type", "Unknown"),
                "visual_confidence": visual.get("confidence", 0),
                "hazard_detected": False,
                "overflow_detected": False,
                "collection_required": True,
                "priority": "HIGH",
                "priority_value": PRIORITIES["HIGH"],
                "action": "COLLECT",
                "reason": "Current fill level reached the collection threshold.",
                "decision_source": "SAFETY_RULE"
            }

        # 8. Return the final decision
        return {
            "bin_id": prediction.get("bin_id"),
            "current_fill": current_fill,
            "predicted_fill": prediction.get("predicted_fill_10min"),
            "waste_type": visual.get("waste_type", "Unknown"),
            "visual_confidence": visual.get("confidence", 0),
            "hazard_detected": False,
            "overflow_detected": False,
            "collection_required": result["collection_required"],
            "priority": result["priority"],
            "priority_value": PRIORITIES[result["priority"]],
            "action": result["action"],
            "reason": result["reason"],
            "decision_source": "GROQ_LLM"
        }
