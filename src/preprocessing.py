import numpy as np
import torch
from PIL import Image


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convertit une image RGB en tenseur [3, 32, 32] avec les pixels entre 0 et 1."""
    pixels = np.array(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(pixels).permute(2, 0, 1).to(torch.float32) / 255.0