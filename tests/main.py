import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs

from algorithms.fuzzy_cmeans import FuzzyCMeans

sns.set()
number_of_clusters = 3
m = 2
iterations = 2000
samples = 200
std = 1
X, _ = make_blobs(n_samples=samples, centers=number_of_clusters,
                  cluster_std=std, random_state=random.randint(0, 200))
fuzzy_cmeans = FuzzyCMeans()
fuzzy_cmeans.fit(X, m, number_of_clusters, iterations=iterations)
centers, u = fuzzy_cmeans.centers, fuzzy_cmeans.u
print("centers:")
print(centers)
plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.7)
plt.show()
point = np.array([[3, 1]])
predicted = fuzzy_cmeans.predict(point)
print()
print(f"predicted = {predicted}")
