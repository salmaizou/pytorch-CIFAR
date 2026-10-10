import kagglehub
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

# 1. Chemin du projet
PROJECT_DIR = Path(__file__).resolve().parent.parent

# 2. Dossier de stockage des données
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 3. Télécharger CIFAR-10 depuis Kaggle
dataset_path = kagglehub.dataset_download(
    "pankrzysiu/cifar10-python",
    output_dir=str(DATA_DIR)
)


def convert_batches_to_images(dataset_dir: Path, output_dir: Path) -> None:
    with (dataset_dir / "batches.meta").open("rb") as file:
        metadata = pickle.load(file, encoding="bytes")

    class_names = [name.decode("utf-8") for name in metadata[b"label_names"]]
    batches = {
        "train": [f"data_batch_{number}" for number in range(1, 6)],
        "test": ["test_batch"],
    }

    for split, batch_names in batches.items():
        image_count = 0
        for batch_name in batch_names:
            with (dataset_dir / batch_name).open("rb") as file:
                batch = pickle.load(file, encoding="bytes")

            for pixels, label, filename in zip(
                batch[b"data"], batch[b"labels"], batch[b"filenames"]
            ):
                class_name = class_names[label]
                class_dir = output_dir / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                image = pixels.reshape(3, 32, 32).transpose(1, 2, 0)
                image_name = Path(filename.decode("utf-8")).name
                Image.fromarray(np.asarray(image, dtype=np.uint8)).save(class_dir / image_name)
                image_count += 1

        print(f"{split}: {image_count} images converties")


dataset_dir = Path(dataset_path)
if not (dataset_dir / "batches.meta").exists():
    dataset_dir = next(path.parent for path in dataset_dir.rglob("batches.meta"))

images_dir = DATA_DIR / "images"
convert_batches_to_images(dataset_dir, images_dir)
print("Images enregistrées dans :", images_dir)

# 4. Afficher le chemin
print("Dataset téléchargé avec succès !")
print("Chemin :", dataset_path)

# 5. Afficher les fichiers du dataset
print("\nFichiers disponibles :")

for file in Path(dataset_path).rglob("*"):
    if file.is_file():
        print(file.relative_to(dataset_path))