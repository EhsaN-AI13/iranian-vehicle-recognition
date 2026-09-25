from pathlib import Path

from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split

from PIL import Image

from torch.utils.data import DataLoader, Dataset

# Dataset
DATA_DIR = Path("train")




# Image Transforms


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
),
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Creating a Dataset
dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=val_transform,
)

print("Number of images:", len(dataset))
print("Number of classes:", len(dataset.classes))
print("Classes:", dataset.classes)
print("Class to index:", dataset.class_to_idx)

image, label = dataset[0]

print("Image shape:", image.shape)
print("Label:", label)
print("Pixel range:", image.min().item(), "to", image.max().item())


from PIL import Image
from torch.utils.data import Dataset


class VehicleDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# The full path to the images and the label corresponding to each image
samples = dataset.samples

image_paths = [sample[0] for sample in samples]
labels = [sample[1] for sample in samples]


# Stratified splitting
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)


print("\nTrain images:", len(train_paths))
print("Validation images:", len(val_paths))


# Create Train / Validation Dataset


train_dataset = VehicleDataset(
    train_paths,
    train_labels,
    transform=train_transform,
)

val_dataset = VehicleDataset(
    val_paths,
    val_labels,
    transform=val_transform,
)


print("\nTrain Dataset:", len(train_dataset))
print("Validation Dataset:", len(val_dataset))


# Examining a Sample
train_image, train_label = train_dataset[0]

print("Train image shape:", train_image.shape)
print("Train label:", train_label)


# DataLoaders


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)


# Reviewing a Batch
images, labels = next(iter(train_loader))

print("\nBatch images shape:", images.shape)
print("Batch labels shape:", labels.shape)
print("First 5 labels:", labels[:5])