import numpy as np
from scipy.linalg import norm


class FuzzyCMeans:
    def __init__(self, number_of_clusters: int, m: int = 2):
        self.number_of_clusters = number_of_clusters
        self.m = m

    def compute(self, X, iterations: int = 1000):
        u = np.random.rand(X.shape[0], self.number_of_clusters)
        u = np.fmax(u, np.finfo(np.float64).eps)
        centers = None
        iteration = 1
        while iteration <= iterations:
            centers = self.compute_next_centers(X, u)
            u = self.compute_next_u(X, centers)
            iteration += 1
        return centers, u

    def compute_next_centers(self, X, u):
        centers = []
        for i in range(self.number_of_clusters):
            temp = np.power(u[:, i].reshape(-1, 1), self.m)
            t = np.sum(temp * X, axis=0)
            temp = t.reshape(-1) / temp.sum()
            centers.append(temp)
        centers = np.asarray(centers)
        centers = np.fmax(centers, np.finfo(np.float64).eps)
        return centers

    def compute_next_u(self, X, centers):
        u = np.zeros((X.shape[0], self.number_of_clusters))
        for i in range(X.shape[0]):
            for j in range(self.number_of_clusters):
                temp = X[i, :].reshape(1, -1)
                t = centers[j, :].reshape(1, -1)
                neom = norm(temp - t)
                denom = temp - centers
                denom = norm(denom, axis=1)
                t = np.power(neom / denom, 2 // (self.m - 1))
                u[i, j] = 1 / t.sum()
        u = np.fmax(u, np.finfo(np.float64).eps)
        return u
