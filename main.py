import numpy as np


def activation_relu(w):
    return np.maximum(0, w)


def cost(y_hat, y):
    return np.square(y_hat - y).sum()


def cost_pd(y_hat, y):
    return 2 * (y_hat - y)


D_N, D_IN, D_H, D_OUT = 64, 1000, 100, 10

X = np.random.randn(D_N, D_IN)
Y = np.random.randn(D_N, D_OUT)

W1 = np.random.randn(D_IN, D_H)
W2 = np.random.randn(D_H, D_OUT)

B1 = np.random.randn(1, D_IN)
B2 = np.random.randn((1, D_H))

learning_rate = 0.000001
for t in range(500):
    H = activation_relu(X.dot(W1) + B1)
    Y_HAT = activation_relu(H.dot(W2) + B2)

    loss = cost(Y_HAT, Y)
    print(t, loss)

    ##Backprop to compute gradients of w1 and w2 with respect to loss
    PD_LOSS = cost_pd(Y_HAT, Y)
