from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
SOURCE_DIR = DATA_DIR / "cifar-10-batches-py"
IMAGES_DIR = DATA_DIR / "images"

print("=== FICHIERS CIFAR-10 ORIGINAUX ===")
if SOURCE_DIR.is_dir():
	for file in sorted(SOURCE_DIR.iterdir()):
		if file.is_file():
			print(file.relative_to(DATA_DIR))
else:
	print(f"Dossier introuvable : {SOURCE_DIR}")

print("\n=== VÉRIFICATION DES IMAGES ===")
for split in ("train", "test"):
	split_dir = IMAGES_DIR / split
	if not split_dir.is_dir():
		print(f"{split}: dossier introuvable")
		continue

	split_count = 0
	for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
		images = sorted(class_dir.glob("*.png"))
		split_count += len(images)
		print(f"{split}/{class_dir.name}: {len(images)} images")
		if images:
			print(f"  Exemple : {images[0].relative_to(DATA_DIR)}")

	print(f"Total {split}: {split_count} images")

print("\n=== AFFICHAGE DE 10 IMAGES AVEC LEURS ÉTIQUETTES ===")
train_dir = IMAGES_DIR / "train"
samples = []
if train_dir.is_dir():
	for class_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
		images = sorted(class_dir.glob("*.png"))
		if images:
			samples.append((images[0], class_dir.name))
		if len(samples) == 10:
			break

if len(samples) < 10:
	print(f"Seulement {len(samples)} images disponibles pour l'affichage.")
else:
	figure, axes = plt.subplots(2, 5, figsize=(15, 6))
	for axis, (image_path, label) in zip(axes.flat, samples):
		with Image.open(image_path) as image:
			axis.imshow(image.convert("RGB"))
		axis.set_title(label)
		axis.axis("off")
	figure.tight_layout()
	plt.show()
