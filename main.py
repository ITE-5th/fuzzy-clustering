import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs

from fuzzy_cmeans import FuzzyCMeans

sns.set()
number_of_clusters = 2
X, _ = make_blobs(n_samples=300, centers=number_of_clusters,
                  cluster_std=0.60, random_state=0)
fuzzy_cmeans = FuzzyCMeans(number_of_clusters=number_of_clusters)
fuzzy_cmeans.fit(X, iterations=100)
centers, u = fuzzy_cmeans.centers, fuzzy_cmean.u
print("centers:")
print(centers)
plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.7)
plt.show()
point = np.array([[3, 1]])
predicted = fuzzy_cmeans.predict(point)
print()
print(f"predicted = {predicted}")
