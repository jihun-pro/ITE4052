import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def gen_gaussian_filter(m: int, sig: int):
    ax = np.linspace(-(m - 1) / 2, (m - 1) / 2, m)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    kernel = np.expand_dims(kernel, -1)
    kernel = np.expand_dims(kernel, -1)
    return kernel / np.sum(kernel)


def main():
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    img = x_train[np.random.randint(0, 50000)].astype('float32')
    img /= 255
    img = np.expand_dims(img, 0)

    plt.imshow(img[0])
    plt.suptitle('Original')
    plt.axis('off')
    fig_output, axs_output = plt.subplots(3, 3)
    fig_output.suptitle('Output')
    fig_filter, axs_filter = plt.subplots(3, 3)
    fig_filter.suptitle('Filter')

    for i_m, m in enumerate([3, 7, 15]):
        for j_m, sig in enumerate([1, 3, 5]):
            filter = gen_gaussian_filter(m, sig)

            output = np.empty((1, img.shape[1], img.shape[2], 0))
            for channel in range(img.shape[3]):
                output = np.append(output,
                                   tf.nn.conv2d(img[:, :, :, channel:channel+1], filter, strides=None, padding="SAME"),
                                   -1)

            axs_output[i_m, j_m].imshow(output[0])
            axs_output[i_m, j_m].axis('off')
            axs_output[i_m, j_m].set_title(f'm = {m}, sig = {sig}')

            axs_filter[i_m, j_m].imshow(filter[:,:,0,0])
            axs_filter[i_m, j_m].axis('off')
            axs_filter[i_m, j_m].set_title(f'm = {m}, sig = {sig}')

    plt.show()


if __name__ == '__main__':
    main()