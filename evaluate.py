import torch
import torch.nn as nn
from torchvision.models import resnet18
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import val_loader, dataset


# Device

device = torch.device("cpu")

print("Device:", device)

# Model


NUM_CLASSES = 29

model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES,
)



# Load Best Model


model.load_state_dict(
    torch.load(
        "models/finetune_best_model.pth",
        map_location=device,
    )
)

model = model.to(device)

model.eval()



# Evaluation


all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )



# Classification Report


print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=dataset.classes,
        zero_division=0,
    )
)



# Confusion Matrix


cm = confusion_matrix(
    all_labels,
    all_predictions,
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)