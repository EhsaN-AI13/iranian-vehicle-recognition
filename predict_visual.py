import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import datasets
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path



# Settings


DEVICE = torch.device("cpu")

MODEL_PATH = "models/finetune_best_model.pth"
IMAGE_PATH = "test_images/car4.jpg"

NUM_CLASSES = 29



# Transform


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])



# Class Names


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
        map_location=DEVICE,
        weights_only=True,
    )
)

model = model.to(DEVICE)
model.eval()



# Load Image


image = Image.open(IMAGE_PATH).convert("RGB")

input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.to(DEVICE)



# Prediction


with torch.no_grad():

    outputs = model(input_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1,
    )

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=3,
        dim=1,
    )



# Results


predicted_index = top_indices[0][0].item()
predicted_class = class_names[predicted_index]
confidence = top_probabilities[0][0].item() * 100


print("\n" + "=" * 50)
print("VEHICLE RECOGNITION RESULT")
print("=" * 50)

print(f"Prediction : {predicted_class}")
print(f"Confidence : {confidence:.2f}%")

print("\nTop 3:")

for i in range(3):

    index = top_indices[0][i].item()
    probability = top_probabilities[0][i].item() * 100

    print(
        f"{i + 1}. "
        f"{class_names[index]} "
        f"→ {probability:.2f}%"
    )



# Display Image


plt.figure(figsize=(8, 6))

plt.imshow(image)

plt.title(
    f"Prediction: {predicted_class}\n"
    f"Confidence: {confidence:.2f}%"
)

plt.axis("off")

plt.tight_layout()

plt.show()