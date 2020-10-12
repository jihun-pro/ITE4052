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
    Ixx_sum = np.zeros_like(Ixx)    # Ixx_sum[i][j] = sum(Ixx[i-r : i+r+1, j-r : j+r+1])
    Ixy_sum = np.zeros_like(Ixy)    # Ixy_sum[i][j] = sum(Ixy[i-r : i+r+1, j-r : j+r+1])
    Iyy_sum = np.zeros_like(Iyy)    # Iyy_sum[i][j] = sum(Iyy[i-r : i+r+1, j-r : j+r+1])
    Ixx_sum[r][r] = np.sum(Ixx[0 : win_size, 0 : win_size])
    Ixy_sum[r][r] = np.sum(Ixy[0 : win_size, 0 : win_size])
    Iyy_sum[r][r] = np.sum(Iyy[0 : win_size, 0 : win_size])

    for i in range(r + 1, img.shape[0] - r):
        Ixx_sum[i][r] = Ixx_sum[i - 1][r] + np.sum(Ixx[i+r, :win_size])
        Ixy_sum[i][r] = Ixy_sum[i - 1][r] + np.sum(Ixy[i+r, :win_size])
        Iyy_sum[i][r] = Iyy_sum[i - 1][r] + np.sum(Iyy[i+r, :win_size])
    for i in range(r + 1, img.shape[1] - r):
        Ixx_sum[r][i] = Ixx_sum[r][i - 1] + np.sum(Ixx[:win_size, i+r])
        Ixy_sum[r][i] = Ixy_sum[r][i - 1] + np.sum(Ixy[:win_size, i+r])
        Iyy_sum[r][i] = Iyy_sum[r][i - 1] + np.sum(Iyy[:win_size, i+r])

    for i in range(r + 1, img.shape[0] - r):
        for j in range(r + 1, img.shape[1] - r):
            H = np.ndarray((2, 2))
            Ixx_sum[i][j] = Ixx_sum[i - 1][j] + Ixx_sum[i][j - 1] - Ixx_sum[i - 1][j - 1] + Ixx[i+r][j+r]
            Ixy_sum[i][j] = Ixy_sum[i - 1][j] + Ixy_sum[i][j - 1] - Ixy_sum[i - 1][j - 1] + Ixy[i+r][j+r]
            Iyy_sum[i][j] = Iyy_sum[i - 1][j] + Iyy_sum[i][j - 1] - Iyy_sum[i - 1][j - 1] + Iyy[i+r][j+r]
            H[0][0] = Ixx_sum[i][j] - Ixx_sum[i - win_size][j] - Ixx_sum[i][j - win_size] + Ixx_sum[i - win_size][j - win_size]
            H[0][1] = H[1][0] = Ixy_sum[i][j] - Ixy_sum[i - win_size][j] - Ixy_sum[i][j - win_size] + Ixy_sum[i - win_size][j - win_size]
            H[1][1] = Iyy_sum[i][j] - Iyy_sum[i - win_size][j] - Iyy_sum[i][j - win_size] + Iyy_sum[i - win_size][j - win_size]
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
