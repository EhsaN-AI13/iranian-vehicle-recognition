from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

train_dir = Path("train")

classes = [
    "206",
    "405",
    "Dena",
    "Pars",
    "Samand",
    "Shahin",
    "Tiba",
    "Quik",
    "Saina",
    "Runna",
    "Pride131nasimsaba",
    "Pride132and111",
]

samples_per_class = 2

fig, axes = plt.subplots(
    len(classes),
    samples_per_class,
    figsize=(10, len(classes) * 3)
)

for row, class_name in enumerate(classes):

    class_dir = train_dir / class_name
    images = list(class_dir.glob("*.jpg"))

    selected = random.sample(
        images,
        min(samples_per_class, len(images))
    )

    for col in range(samples_per_class):

        ax = axes[row, col]

        if col < len(selected):
            image_path = selected[col]

            with Image.open(image_path) as img:
                ax.imshow(img)

            ax.set_title(class_name)

        ax.axis("off")

plt.tight_layout()
plt.show()