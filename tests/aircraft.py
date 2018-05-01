import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from file_path_manager import FilePathManager


def get_type(s):
    try:
        ind = s.index(" ")
        return s[:ind]
    except:
        return s


np.random.seed(5)

X = pd.read_csv(FilePathManager.resolve("data/aircraft.csv"), ",")
# types = X["Aircraft"]
# types = types.apply(get_type)
X = X.as_matrix()[:, 1:]
pca = PCA(2)
X = pca.fit_transform(X)
kmeans = KMeans(4)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
print(kmeans.labels_)
plt.scatter(X[:, 0], X[:, 1], edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", edgecolor='k')
plt.show()
