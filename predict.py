import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import datasets
from PIL import Image
from pathlib import Path

from src.dataset import val_transform



# Settings


device = torch.device("cpu")

NUM_CLASSES = 29

MODEL_PATH = "models/finetune_best_model.pth"
IMAGE_DIR = Path("test_images")



# Load Classes


class_dataset = datasets.ImageFolder("train")
class_names = class_dataset.classes


# Load Model


model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES,
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)

model = model.to(device)
model.eval()



# Prediction Function


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = val_transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=3,
            dim=1,
        )

    print("\n" + "=" * 60)
    print(f"Image: {image_path.name}")
    print("=" * 60)

    for i in range(3):

        class_index = top_indices[0][i].item()

        probability = top_probabilities[0][i].item()

        class_name = class_names[class_index]

        print(
            f"{i + 1}. "
            f"{class_name:<20}"
            f"→ {probability * 100:.2f}%"
        )



# Predict All Images


images = list(IMAGE_DIR.glob("*.jpg"))

print(f"Found {len(images)} images.")

for image_path in images:

    predict_image(image_path)