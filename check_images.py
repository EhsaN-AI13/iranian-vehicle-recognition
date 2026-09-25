from pathlib import Path
from PIL import Image
from collections import Counter

train_dir = Path("train")

sizes = Counter()
corrupted = []

for image_path in train_dir.rglob("*.jpg"):
    try:
        with Image.open(image_path) as img:
            sizes[img.size] += 1
    except Exception:
        corrupted.append(image_path)

print("Total images:", sum(sizes.values()))
print("\nImage sizes:")

for size, count in sizes.most_common():
    print(f"{size}: {count}")

print("\nCorrupted images:", len(corrupted))

if corrupted:
    print("\nFirst corrupted files:")
    for path in corrupted[:10]:
        print(path)