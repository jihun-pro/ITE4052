import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np


def compute_low_eigval(img: np.ndarray, win_size: int):
    low_eigval = np.zeros_like(img)
    Iy, Ix = np.gradient(img)
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    r = win_size // 2
    for i in range(r, img.shape[0] - r):
        for j in range(r, img.shape[1] - r):
            H = np.ndarray((2, 2))
            H[0][0] = np.sum(Ixx[i-r : i+r+1, j-r : j+r+1])
            H[0][1] = H[1][0] = np.sum(Ixy[i-r : i+r+1, j-r : j+r+1])
            H[1][1] = np.sum(Iyy[i-r : i+r+1, j-r : j+r+1])
            low_eigval[i][j] = np.min(np.linalg.eigvals(H))

    return low_eigval


def main(img_path: str):
    img = load_img(img_path, color_mode='grayscale')
    img = img_to_array(img).squeeze(-1)
    img = img.astype('float32') / 255
    for ws in [3, 7, 11]:
        result = compute_low_eigval(img, ws)
        result = array_to_img(np.expand_dims(result, -1))
        result.show()


if __name__ == '__main__':
    main('data/checkerboard.png')
