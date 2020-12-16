import numpy as np
from PIL import Image


# do not import any other modules

# pre-defined util
def show_np_arr(np_arr):  # np_arr should be 2-dim
    tmp = np_arr
    tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
    tmp = np.clip(255 * tmp, 0, 255)
    tmp = Image.fromarray(np.uint8(tmp)).convert('RGB')
    tmp.show()


# load and pre-process dataset, do not modify here
x_train = np.load("./x_train.npy") / 255.
y_train = np.load("./y_train.npy")
x_test = x_train[2]

idx = np.argsort(y_train)
x_train = x_train[idx]
x_train = x_train[::200]
print(x_train.shape, x_test.shape)

# Q1. Eigenface
##step1: compute mean of x_train
mu = np.mean(x_train, 0)
show_np_arr(mu)

##step2: subtract the mean
phi = x_train - mu
phi = np.reshape(phi, (phi.shape[0], -1))

##step3: compute covariance C
C = np.cov(phi.T)

# step4: Compute eigenvector of C, you don't need to do anything at step4.
eigenvalues, eigenvec = np.linalg.eig(C)
eigenvec = eigenvec.T
print("Shape of eigen vectors = ", eigenvec.shape)

##step5: choose K
K = 40

##step6: show top K eigenfaces. use show_np_arr func.

canvas = np.zeros((5 * 28, 8 * 28))
for i in range(K):
    eigenface = np.reshape(eigenvec[i], (28, 28))
    eigenface = (eigenface - np.min(eigenface)) / (np.max(eigenface) - np.min(eigenface))
    eigenface = np.clip(255 * eigenface, 0, 255)
    canvas[i // 8 * 28: i // 8 * 28 + 28, i % 8 * 28: i % 8 * 28 + 28] = eigenface

show_np_arr(canvas)

# Q2. Image Approximation
x = x_test

##step1: approximate x as x_hat with top K eigenfaces and show x_hat
x_hat = mu.copy()
K = 0
eig_sum = 0

while eig_sum < np.sum(eigenvalues) * 1:
    w = np.dot(eigenvec[K], (x - mu).reshape(-1, 1))
    eigenface = np.reshape(eigenvec[K], (28, 28))
    x_hat += (w * eigenface).astype('float')

    eig_sum += eigenvalues[K]
    K += 1

show_np_arr(x)
show_np_arr(x_hat)

##step2: compater mse between x and x_hat by changing the number of the eigenfaces used for reconstruction (approximation) from 1 to K
x_hat = mu.copy()
for i in range(0, K + 1):
    w = np.dot(eigenvec[i].T, (x - mu).reshape(-1, 1))
    eigenface = np.reshape(eigenvec[i], (28, 28))
    x_hat += (w * eigenface).astype('float')
    mse = np.mean(np.square(x - x_hat))
    print(i + 1, mse)




# Q3. Implement fast version of you algorithm in Q1. Show top 'K' eigenfaces using show_np_arr(...)
print('\n============================\n')
mu = np.mean(x_train, 0)
phi = x_train - mu
phi = np.reshape(phi, (phi.shape[0], -1))
C = np.cov(phi)
eigenvalues, eigenvec = np.linalg.eig(C)
eigenvec = phi.T.dot(eigenvec).T
eigenvec /= np.linalg.norm(eigenvec, axis=1).reshape(-1, 1)

x_hat = mu.copy()
K = 0
eig_sum = 0

while eig_sum < np.sum(eigenvalues) * 0.85:
    w = np.dot(eigenvec[K], (x - mu).reshape(-1, 1))
    eigenface = np.reshape(eigenvec[K], (28, 28))
    x_hat += (w * eigenface).astype('float')
    mse = np.mean(np.square(x - x_hat))
    print(K + 1, mse)

    eig_sum += eigenvalues[K]
    K += 1

show_np_arr(x_hat)
