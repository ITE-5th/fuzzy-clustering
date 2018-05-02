import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs

from algorithms.fuzzy_cmeans import FuzzyCMeans
from algorithms.gk import GK

sns.set()
number_of_clusters = 5
m = 2
iterations = 2000
samples = 200
std = 1
X, _ = make_blobs(n_samples=samples, centers=number_of_clusters,
                  cluster_std=std, random_state=random.randint(0, 200))
fuzzy_cmeans = FuzzyCMeans()
gk = GK(number_of_clusters)
gk.fit(X)
centers = gk.centers

print("centers:")
print(centers)
plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.7)
plt.show()
