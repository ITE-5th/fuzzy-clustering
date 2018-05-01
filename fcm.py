import numpy as np
from scipy.linalg import norm


class FCM:
    def __init__(self, n_cluster=4, max_iter=100, m=2, error=1e-6):
        super().__init__()
        self.u, self.centers, self.m = None, None, None
        self.clusters_count = n_cluster
        self.max_iter = max_iter
        self.m = m
        self.error = error

    def fit(self, X):
        N = X.shape[0]
        C = self.clusters_count
        centers = []

        u = np.random.dirichlet(np.ones(C), size=N)

        iteration = 0
        while iteration < self.max_iter:
            u2 = u.copy()

            centers = self.next_centers(X, u)
            u = self.next_u(X, centers)
            iteration += 1

            # Stopping rule
            if norm(u - u2) < self.error:
                break

        self.u = u
        self.centers = centers
        return centers

    def next_centers(self, X, u):
        um = u ** self.m
        return (X.T @ um / np.sum(um, axis=0)).transpose()

    def next_u(self, X, centers):
        return np.apply_along_axis(self._predict, 1, X, centers)

    def _predict(self, X, centers):
        power = float(2 / (self.m - 1))
        x1 = norm(X - centers, axis=1) ** power
        denominator_ = x1.reshape((1, -1)).repeat(x1.shape[0], axis=0)
        denominator_ = x1[:, None] / denominator_

        return 1 / denominator_.sum(1)

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        u = np.apply_along_axis(self._predict, 1, X, self.centers)

        # for i in range(len(X)):
        #     u.append(self._predict(X[i], self.centers))
        return np.argmax(u)
