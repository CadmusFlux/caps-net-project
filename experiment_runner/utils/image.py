import cv2
import numpy as np


def load_image(location: str) -> np.ndarray:
    image = cv2.imread(location)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
