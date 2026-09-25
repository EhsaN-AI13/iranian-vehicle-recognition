import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path

from src.dataset import train_loader, val_loader



# Device


device = torch.device("cpu")

print("Device:", device)



# Model


NUM_CLASSES = 29

model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)



# Freeze Backbone


for param in model.parameters():
    param.requires_grad = False



# Unfreeze Layer 4


for param in model.layer4.parameters():
    param.requires_grad = True



# Replace Classifier


model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES,
)


# Classifier must be trainable
for param in model.fc.parameters():
    param.requires_grad = True


model = model.to(device)



# Trainable Parameters


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
print(
    "Frozen parameters:",
    total_params - trainable_params
)



# Loss


criterion = nn.CrossEntropyLoss()



# Optimizer


optimizer = Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=0.0001,
)



# Training Settings


EPOCHS = 5

checkpoint_path = Path(
    "models/finetune_checkpoint.pth"
)

best_model_path = Path(
    "models/finetune_best_model.pth"
)

start_epoch = 0
best_val_accuracy = 0.0



# Load Fine-Tuning Checkpoint


if checkpoint_path.exists():

    print("\nLoading fine-tuning checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    best_val_accuracy = checkpoint[
        "val_accuracy"
    ]

    print(
        f"Resuming from epoch {start_epoch}"
    )

    print(
        f"Previous validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )



# Training Loop


for epoch in range(
    start_epoch,
    EPOCHS
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    
    # Training
    

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


        # Statistics

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss /
        len(train_loader)
    )

    train_accuracy = (
        correct /
        total
    )


    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.4f}"
    )


    
    # Validation
    

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_loss /
        len(val_loader)
    )

    val_accuracy = (
        correct /
        total
    )


    print(
        f"Validation Loss: {val_loss:.4f} "
        f"Validation Accuracy: "
        f"{val_accuracy:.4f}"
    )


    
    # Save Checkpoint
    

    checkpoint = {

        "epoch": epoch + 1,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,
    }


    torch.save(
        checkpoint,
        checkpoint_path
    )


    print("Fine-tune checkpoint saved.")


    
    # Save Best Model
    

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            "Fine-tune best model saved."
        )


print("\nFine-tuning finished.")