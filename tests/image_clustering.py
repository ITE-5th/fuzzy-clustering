import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import ndimage
from sklearn import cluster

from algorithms.fcm import FCM
from file_path_manager import FilePathManager

sns.set()

image = ndimage.imread(FilePathManager.resolve("images/car.jpg"))
plt.figure(figsize=(15, 8))
plt.imshow(image)

x, y, z = image.shape
image_2d = image.reshape(x * y, z)

kmeans_cluster = cluster.KMeans(n_clusters=7)
kmeans_cluster.fit(image_2d)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_labels = kmeans_cluster.labels_

plt.figure(figsize=(15, 8))
plt.title("KMeans")
plt.imshow(cluster_centers[cluster_labels].astype(np.int32).reshape(x, y, z))

cmeans_cluster = FCM(n_clusters=7)
cmeans_cluster.fit(image_2d)
cluster_centers = cmeans_cluster.centers
cluster_labels = np.arange(0, 7)

output = cmeans_cluster.predict(image_2d)

plt.figure(figsize=(15, 8))
plt.title("CMeans")
plt.imshow(cluster_centers[output].astype(np.int32).reshape(x, y, 3))
plt.show()
