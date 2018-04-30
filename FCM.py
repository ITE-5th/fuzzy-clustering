import numpy as np
from scipy.linalg import norm


class FCM:
    def __init__(self, n_cluster=4, max_iter=100, m=2):
        super().__init__()
        self.u, self.centers, self.m = None, None, None
        self.clusters_count = n_cluster
        self.max_iter = max_iter
        self.m = m

    def fit(self, X):
        N = X.shape[0]
        C = self.clusters_count
        centers = []

        u = np.random.dirichlet(np.ones(C), size=N)

        iteration = 0
        while iteration < self.max_iter:
            centers = self.next_centers(X, u)
            u = self.next_u(X, centers)
            iteration += 1

        self.u = u
        self.centers = centers
        return centers

    def next_centers(self, X, u):
        clusters = []

        for j in range(self.clusters_count):
            numerator = 0
            for i in range(X.shape[0]):
                numerator += (u[i, j] ** self.m) * X[i]

            denominator = np.sum(u[:, j] ** self.m)
            clusters.append(numerator / denominator)

        return np.asarray(clusters)

    def next_u(self, X, centers):
        N = X.shape[0]
        C = self.clusters_count
        new_u = np.zeros((N, C))

        for i in range(N):
            new_u[i] = self._predict(X[i], centers)
        return new_u

    def _predict(self, X, centers):
        power = float(2 / (self.m - 1))
        u = np.zeros(self.clusters_count)

        denominator_ = norm(X - centers, axis=1)
        for k in range(self.clusters_count):
            numerator_ = norm(X - centers[k])
            denominator = np.power((numerator_ / denominator_), power)
            u[k] = 1 / denominator.sum()

        return u

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        u = []
        for i in range(len(X)):
            u.append(self._predict(X[i], self.centers))
        return np.argmax(u)
