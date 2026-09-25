import torch.nn as nn
from torchvision.models import resnet18


NUM_CLASSES = 29



# Load Pretrained ResNet18


model = resnet18(weights="DEFAULT")



# Freeze Backbone


for param in model.parameters():
    param.requires_grad = False



# Replace Classifier


model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES,
)


# Only the classifier should be trained.
for param in model.fc.parameters():
    param.requires_grad = True



# Check Trainable Parameters


trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)


print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)
print("Frozen parameters:", total_params - trainable_params)

print("\nClassifier:")
print(model.fc)