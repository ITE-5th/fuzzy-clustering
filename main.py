import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs

from fuzzy_cmean import FuzzyCMeans

sns.set()

X, _ = make_blobs(n_samples=300, centers=3,
                  cluster_std=0.60, random_state=0)
fuzzy_cmean = FuzzyCMeans(number_of_clusters=3)
centers, u = fuzzy_cmean.fit(X)
print(centers)
plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
