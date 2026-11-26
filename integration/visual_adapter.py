
"""
Adapter connecting the teammate's Visual Agent
to our Decision Agent.
"""

from visual_agent.vision_agent import load_model, predict_waste


class VisualAdapter:

    def __init__(self, weights_path):
        self.model = load_model(weights_path)

    def analyze(self, image_path):

        waste_type, confidence = predict_waste(
            image_path,
            self.model
        )

        return {
            "waste_type": waste_type,
            "confidence": confidence,

            # Hazard detection is not implemented
            # by the six-class CNN.
            "hazard_detected": False
        }
