import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs

from fuzzy_cmean import FuzzyCMeans

sns.set()
number_of_clusters = 2
X, _ = make_blobs(n_samples=300, centers=number_of_clusters,
                  cluster_std=0.60, random_state=0)
fuzzy_cmean = FuzzyCMeans(number_of_clusters=number_of_clusters)
our_centers, our_u = fuzzy_cmean.compute(X, iterations=100)
# skfuzzy_centers, skfuzzy_u, _, _, _, _, _ = cmeans(X.T, number_of_clusters, 2, error=0.005, maxiter=1000, init=None)
# skfuzzy_centers = skfuzzy_centers.T
print("centers:")
print(our_centers)
# print("theirs:")
# print(skfuzzy_centers)
plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
plt.scatter(our_centers[:, 0], our_centers[:, 1], c='black', s=200, alpha=0.7)
# plt.scatter(skfuzzy_centers[:, 0], skfuzzy_centers[:, 1], c='red', s=200, alpha=0.7)
plt.show()
