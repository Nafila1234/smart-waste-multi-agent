"""Vision agent: classifies a waste photo into one of 6 classes."""
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CLASSES = ["battery", "glass", "metal", "organic", "paper", "plastic"]
_device = "cuda" if torch.cuda.is_available() else "cpu"
_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_model(weights_path):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(torch.load(weights_path, map_location=_device))
    return m.to(_device).eval()

def predict_waste(image, model):
    """image: PIL image or file path. Returns (class_name, confidence)."""
    if isinstance(image, str):
        image = Image.open(image)
    x = _tf(image.convert("RGB")).unsqueeze(0).to(_device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    i = int(probs.argmax())
    return CLASSES[i], float(probs[i])
