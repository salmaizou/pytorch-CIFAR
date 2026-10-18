from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .preprocessing import preprocess_image


PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_DIR / "data" / "images"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class CIFAR10ImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, images_dir: Path, class_to_idx: dict[str, int] | None = None) -> None:
        if not images_dir.is_dir():
   
   
            raise FileNotFoundError(f"Le dossier d'images est introuvable : {images_dir}")

        if class_to_idx is None:
            classes = sorted(path.name for path in images_dir.iterdir() if path.is_dir())
            if not classes:
                raise FileNotFoundError(f"Aucun dossier de classe trouvé dans : {images_dir}")
            self.class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
        else:
            actual_classes = {
                path.name for path in images_dir.iterdir() if path.is_dir()
            }
            expected_classes = set(class_to_idx)
            if actual_classes != expected_classes:
                raise ValueError(
                    f"Les classes de {images_dir} ne correspondent pas à celles de train."
                )
            self.class_to_idx = class_to_idx

        self.classes = list(self.class_to_idx)
        self.samples: list[tuple[Path, int]] = []
        for class_name, class_index in self.class_to_idx.items():
            class_dir = images_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Le dossier de classe est introuvable : {class_dir}")

            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, class_index))

        if not self.samples:
            raise FileNotFoundError(f"Aucune image trouvée dans : {images_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image_tensor = preprocess_image(image)
        return image_tensor, label


def create_dataloaders(
    data_dir: Path = IMAGES_DIR,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
]:
    """Crée les DataLoaders d'entraînement et de test pour CIFAR-10."""
    if batch_size <= 0:
        raise ValueError("batch_size doit être supérieur à zéro.")
    if num_workers < 0:
        raise ValueError("num_workers ne peut pas être négatif.")

    train_dataset = CIFAR10ImageDataset(data_dir / "train")
    test_dataset = CIFAR10ImageDataset(data_dir / "test", train_dataset.class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader