import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image


class VehiclePredictor:

    def __init__(
        self,
        model_path="models/finetune_best_model.pth",
    ):

        self.device = torch.device("cpu")

        self.num_classes = 29

        self.class_names = [
            "206",
            "207i",
            "405",
            "Arisun",
            "Dena",
            "HcCross",
            "JackS5",
            "KaraMazdaPickup",
            "L90",
            "MVM315H",
            "MVMX22",
            "NeissanVanet",
            "Pars",
            "PeykanSavari",
            "PeykanVanet",
            "Pride131nasimsaba",
            "Pride132and111",
            "Pride141",
            "PrideVanet151",
            "Quik",
            "RenaultPK",
            "RioSD",
            "Runna",
            "Saina",
            "Samand",
            "SamandSoren",
            "Shahin",
            "Tiba",
            "Xantia",
        ]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.model = resnet18(weights=None)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            self.num_classes,
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path, top_k=3):

        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():

            outputs = self.model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            top_probabilities, top_indices = torch.topk(
                probabilities,
                k=top_k,
                dim=1,
            )

        results = []

        for i in range(top_k):

            class_index = top_indices[0][i].item()
            probability = top_probabilities[0][i].item()

            results.append({
                "class": self.class_names[class_index],
                "confidence": probability,
            })

        return results