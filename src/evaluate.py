import torch
from torch import nn

from .dataloader import create_dataloaders
from .model import CIFAR10CNN
from .train import MODEL_PATH


def evaluate_model(batch_size: int = 64) -> tuple[float, float]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modèle entraîné introuvable : {MODEL_PATH}. Lance d'abord src.train."
        )

    _, test_loader = create_dataloaders(batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR10CNN().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            scores = model(images)
            loss = loss_function(scores, labels)

            batch_size_actual = labels.size(0)
            total_loss += loss.item() * batch_size_actual
            total_correct += (scores.argmax(dim=1) == labels).sum().item()
            total_images += batch_size_actual

    average_loss = total_loss / total_images
    accuracy = total_correct / total_images
    print(f"Évaluation sur : {device}")
    print(f"Images de test : {total_images}")
    print(f"Perte test : {average_loss:.4f}")
    print(f"Précision test : {accuracy:.2%}")
    return average_loss, accuracy


if __name__ == "__main__":
    evaluate_model()