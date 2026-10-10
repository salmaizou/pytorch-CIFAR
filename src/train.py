from pathlib import Path

import torch
from torch import nn

from .dataloader import create_dataloaders
from .model import CIFAR10CNN


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "cifar10_cnn.pth"


def train_model(
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
) -> CIFAR10CNN:
    if epochs <= 0:
        raise ValueError("epochs doit être supérieur à zéro.")
    if learning_rate <= 0:
        raise ValueError("learning_rate doit être supérieur à zéro.")

    train_loader, _ = create_dataloaders(batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR10CNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Entraînement sur : {device}")
    print("Pixels mis à l'échelle entre 0 et 1 (division par 255)")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_images = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            scores = model(images)
            loss = loss_function(scores, labels)
            loss.backward()
            optimizer.step()

            batch_size_actual = labels.size(0)
            total_loss += loss.item() * batch_size_actual
            total_correct += (scores.argmax(dim=1) == labels).sum().item()
            total_images += batch_size_actual

        average_loss = total_loss / total_images
        accuracy = total_correct / total_images
        print(
            f"Époque {epoch + 1}/{epochs} - "
            f"perte : {average_loss:.4f} - "
            f"précision entraînement : {accuracy:.2%}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modèle enregistré dans : {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_model()