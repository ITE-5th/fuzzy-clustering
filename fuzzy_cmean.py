import numpy as np
from scipy.linalg import norm


class FuzzyCMeans:
    def __init__(self, number_of_clusters: int, max_iterations: int = 1000, epsilon: float = 0.1):
        self.number_of_clusters = number_of_clusters
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.u = self.centers = None

    def fit(self, X):
        rows, cols = X.shape
        u = np.random.randn(rows, self.number_of_clusters)
        new_u = np.full((rows, self.number_of_clusters), 2)
        iteration = 1
        while iteration <= self.max_iterations and self.stop_condition(u, new_u):
            centers = self.compute_next_centers(X, u)
            u = new_u
            new_u = self.compute_next_u(X, centers)
            iteration += 1
        self.u = u
        self.centers = centers

    def compute_next_centers(self, X, u):
        centers = []
        for i in range(self.number_of_clusters):
            temp = np.sum(u[:, i].reshape(-1, 1) * X, axis=0) / np.sum(u[:, i]).reshape(-1)
            centers.append(temp)
        centers = np.asarray(centers)
        print("centers:")
        print(centers)
        return centers

    def compute_next_u(self, X, centers):
        new_u = np.zeros((X.shape[0], self.number_of_clusters))
        for i in range(X.shape[0]):
            for j in range(self.number_of_clusters):
                temp = X[i, :].reshape(1, -1)
                t = centers[j, :].reshape(1, -1)
                neom = norm(temp - t)
                denom = norm(temp - centers)
                new_u[i, j] = 1 / np.power(neom / denom, 2).sum()
        return new_u

    def stop_condition(self, u, new_u):
        return abs(u.max() - new_u.max()) > self.epsilon
