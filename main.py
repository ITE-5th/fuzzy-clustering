import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs

from fuzzy_cmean import FuzzyCMeans

sns.set()

X, _ = make_blobs(n_samples=300, centers=3,
                  cluster_std=0.60, random_state=0)
fuzzy_cmean = FuzzyCMeans(number_of_clusters=3)
fuzzy_cmean.fit(X)
centers, u = fuzzy_cmean.centers, fuzzy_cmean.u
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()
