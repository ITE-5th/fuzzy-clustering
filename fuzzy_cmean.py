import numpy as np
from scipy.linalg import norm


class FuzzyCMeans:
    def __init__(self, number_of_clusters: int, iterations: int = 1000, m: int = 2):
        self.number_of_clusters = number_of_clusters
        self.iterations = iterations
        self.u = self.centers = None
        self.m = m

    def fit(self, X):
        rows, cols = X.shape
        u = np.random.uniform(0, 1, (rows, self.number_of_clusters))
        iteration = 1
        while iteration <= self.iterations:
            centers = self.compute_next_centers(X, u)
            u = self.compute_next_u(X, centers)
            iteration += 1
        return centers, u

    def compute_next_centers(self, X, u):
        centers = []
        for i in range(self.number_of_clusters):
            temp = np.sum(u[:, i].reshape(-1, 1) * X, axis=0) / np.sum(u[:, i]).reshape(-1)
            temp = np.fmax(temp, np.finfo(np.float64).eps)
            centers.append(temp)
        centers = np.asarray(centers)
        return centers

    def compute_next_u(self, X, centers):
        u = np.zeros((X.shape[0], self.number_of_clusters))
        for i in range(X.shape[0]):
            for j in range(self.number_of_clusters):
                temp = X[i, :].reshape(1, -1)
                t = centers[j, :].reshape(1, -1)
                neom = norm(temp - t)
                denom = norm(temp - centers)
                u[i, j] = 1 / np.power(neom / denom, 2).sum()
        u = np.fmax(u, np.finfo(np.float64).eps)
        return u
