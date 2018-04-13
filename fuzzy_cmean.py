from functools import partial

import numpy as np
from scipy.linalg import norm


class FuzzyCMeans:
    def __init__(self, number_of_clusters: int, m: int = 2):
        self.number_of_clusters = number_of_clusters
        self.m = m
        self.centers = None
        self.u = None

    def fit(self, X, iterations: int = 1000):
        old_u = np.random.rand(X.shape[0], self.number_of_clusters)
        old_u = np.fmax(old_u, np.finfo(np.float64).eps)
        self.u = np.random.rand(X.shape[0], self.number_of_clusters)
        self.u = np.fmax(self.u, np.finfo(np.float64).eps)
        iteration = 1
        epsilon = 0.1
        while iteration <= iterations and np.abs(self.u - old_u).sum() > epsilon:
            self.centers = self.compute_centers(X)
            old_u = self.u
            self.u = self.compute_u(X)
            iteration += 1

    def predict(self, X):
        return np.apply_along_axis(self.compute_u_for_row, axis=1, arr=X)

    def compute_centers(self, X: np.ndarray):
        fun = partial(self.calculate_multiply, arr=X)
        neo = np.apply_along_axis(fun, 0, self.u)
        dem = self.u.sum(axis=0)
        result = neo / dem
        result = np.fmax(result, np.finfo(np.float64).eps)
        return result

    def compute_u(self, X):
        u = np.zeros((X.shape[0], self.number_of_clusters))
        for i in range(X.shape[0]):
            u[i] = self.compute_u_for_row(X[i])
        u = np.fmax(u, np.finfo(np.float64).eps)
        return u

    def compute_u_for_row(self, row):
        row = row.reshape(1, -1)
        temp = (row - self.centers)
        temp = norm(temp, axis=1)
        temp = temp.reshape(1, -1)
        temp = np.power(temp, 2 / (self.m - 1))
        t = temp.sum()
        result = temp / t
        result = 1 / result
        result = np.fmax(result, np.finfo(np.float64).eps)
        return result.reshape(-1)

    @staticmethod
    def calculate_multiply(col, arr):
        col = col.reshape(arr.shape[0], -1)
        temp = np.sum(col * arr, axis=0)
        return temp
