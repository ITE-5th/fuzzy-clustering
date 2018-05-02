import numpy as np
from scipy.linalg import det, inv, norm


class GK:
    def __init__(self, n_clusters=4, max_iter=100, m=2, error=1e-6):
        super().__init__()
        self.u, self.centers = None, None
        self.clusters_count = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error

    def fit(self, z):
        N = z.shape[0]
        C = self.clusters_count
        centers = []

        u = np.random.dirichlet(np.ones(N), size=C)

        iteration = 0
        while iteration < self.max_iter:
            u2 = u.copy()

            centers = self.next_centers(z, u)
            F = self._covariance(z, centers, u)
            dist = self._distance(z, centers, F)
            u = self.next_u(dist)
            iteration += 1

            # Stopping rule
            if norm(u - u2) < self.error:
                break

        self.u = u
        self.centers = centers
        return centers

    def next_centers(self, z, u):
        um = u ** self.m
        N = len(z)
        C = self.clusters_count

        v = []
        for i in range(C):
            numerator = 0
            denominator = um[i].sum()

            for k in range(N):
                numerator += um[i, k] * z[k]

            v.append(numerator / denominator)

        return np.array(v)
        # return X.T @ um.T / np.sum(um, axis=1)

    def _covariance(self, z, v, u):
        cov = []
        N = len(z)
        C = self.clusters_count

        um = u ** self.m
        for i in range(C):
            denominator = um[i].sum()
            numerator = 0
            for k in range(N):
                # todo: v[:, i] or v[i, :]
                d = np.expand_dims(z[k] - v[i], axis=1)
                d = np.matmul(d, d.transpose())
                numerator += um[i, k] * d

            cov.append(numerator / denominator)

        return np.asarray(cov)

    def _distance(self, z, v, F):
        C = self.clusters_count
        N = len(z)
        d = np.zeros((C, N))

        for i in range(C):
            for k in range(N):
                temp = (z[k] - v[i]).reshape(-1, 1)
                temp2 = np.power(det(F[i]), 1 / self.m) * inv(F[i])
                temp2 = np.matmul(temp.transpose(), temp2)
                d[i, k] = np.matmul(temp2, temp)

        return np.fmax(d, 1e-8)

    def next_u(self, d):
        return self._predict(d)

    def _predict(self, d):
        C, N = d.shape

        power = float(1 / (self.m - 1))
        u = np.zeros((C, N))

        for i in range(C):
            for k in range(N):
                denominator = np.power((d[i, k] / d[:, k]), power)
                u[i, k] = 1 / denominator.sum()

        return u
