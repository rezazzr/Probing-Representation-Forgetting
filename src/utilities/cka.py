import math

import numpy as np
import torch


def hsic(matrix_x: np.ndarray, matrix_y: np.ndarray) -> float:
    n = matrix_x.shape[0]
    matrix_h = np.identity(n) - (1.0 / n) * np.ones((n, n))

    x_times_h = np.matmul(matrix_x, matrix_h)
    y_times_h = np.matmul(matrix_y, matrix_h)

    return 1.0 / ((n - 1) ** 2) * np.trace(np.matmul(x_times_h, y_times_h))


def linear_cka(matrix_x: np.ndarray, matrix_y: np.ndarray) -> float:
    # First center the columns
    matrix_x = matrix_x - np.mean(matrix_x, 0)
    matrix_y = matrix_y - np.mean(matrix_y, 0)

    matrix_x = np.matmul(matrix_x, matrix_x.T)
    matrix_y = np.matmul(matrix_y, matrix_y.T)

    matrix_h = hsic(matrix_x=matrix_x, matrix_y=matrix_y)
    matrix_x = np.sqrt(hsic(matrix_x=matrix_x, matrix_y=matrix_x))
    matrix_y = np.sqrt(hsic(matrix_x=matrix_y, matrix_y=matrix_y))
    return matrix_h / (matrix_x * matrix_y)


class TorchCKA(object):
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        identity = torch.eye(n, device=self.device)
        H = identity - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)
