import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from skfuzzy import cmeans
from sklearn.cluster import KMeans

from algorithms.fcm import FCM
from algorithms.gk import GK

sns.set()

from file_path_manager import FilePathManager

dataset_path = FilePathManager.resolve("data/crime_data.csv")
data = pd.read_csv(dataset_path)
X = data.iloc[:, 2:4].values

number_of_clusters = 4
MAX_ITER = 50
m = 2.00

cntr, _, _, _, _, _, _ = cmeans(X.transpose(), number_of_clusters, m, 1e-8, maxiter=MAX_ITER)
fcm = FCM(number_of_clusters, MAX_ITER, m)
gk = GK(n_clusters=number_of_clusters, m=m, max_iter=MAX_ITER)
kmeans = KMeans(n_clusters=number_of_clusters, max_iter=MAX_ITER)
cmean_centers = fcm.fit(X)
gk_centers = gk.fit(X)
kmeans.fit(X)

points_s = plt.scatter(X[:, 0], X[:, 1], s=100, cmap='viridis')
fcm_s = plt.scatter(cmean_centers[:, 0], cmean_centers[:, 1], c='black', s=250, alpha=0.7)
gk_s = plt.scatter(gk_centers[:, 0], gk_centers[:, 1], c='yellow', s=200, alpha=0.7)

kmeans_centers = kmeans.cluster_centers_
km_s = plt.scatter(kmeans_centers[:, 0], kmeans_centers[:, 1], c='red', s=200, alpha=0.7)

cm_s = plt.scatter(cntr[:, 0], cntr[:, 1], c='green', s=200, alpha=0.7)
plt.legend((gk_s, fcm_s, cm_s, km_s, points_s),
           ("GK Centers", "Our FCM Centers", "skfuzzy FCM Centers", "K-Means Centers", "Points"))
plt.show()
