import os
import cv2
import numpy as np
import glob as gb

CODE = {
    'NORMAL': 0,
    'PNEUMONIA': 1
}


def get_class_name(n):
    for name, label in CODE.items():
        if n == label:
            return name
    return None


def load_images(data_dir: str, img_size=(128, 128)):
    """
    Load chest X-ray images.

    Returns:
        X : np.ndarray -> (N,64,64,3)
        y : np.ndarray -> (N,)
    """

    X = []
    y = []

    for folder in os.listdir(data_dir):

        if folder not in CODE:
            continue

        files = gb.glob(os.path.join(data_dir, folder, "*.jpeg"))

        for file in files:

            image = cv2.imread(file)

            if image is None:
                continue

            image = cv2.resize(image, img_size)

            X.append(image)
            y.append(CODE[folder])

    return np.array(X), np.array(y)


if __name__ == "__main__":

    import sys

    if len(sys.argv) > 1:

        X, y = load_images(sys.argv[1])

        print(f"Loaded {len(X)} images")
        print("Shape:", X.shape)
        print("Labels:", np.bincount(y.astype(int)))

    else:
        print("Usage:")
        print("python preprocess.py <dataset_path>")