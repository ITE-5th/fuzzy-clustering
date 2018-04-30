import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from skfuzzy import cmeans, cmeans_predict
from sklearn.cluster import KMeans

from fcm import FCM

sns.set()

dataset_path = "./crime_data.csv"
data = pd.read_csv(dataset_path)
X = data.iloc[:, 2:].values

number_of_clusters = 4
MAX_ITER = 500
m = 2.00
error = 1e-8

cntr = cmeans(X.transpose(), number_of_clusters, m, error, maxiter=MAX_ITER)[0]
fcm = FCM(number_of_clusters, MAX_ITER, m)
kmeans = KMeans(n_clusters=number_of_clusters, max_iter=MAX_ITER)
cmean_centers = fcm.fit(X)
kmeans.fit(X)
Y = data.iloc[:, 1].values

kmeans_accuracy = 0
fcm_accuracy = 0
cmeans_accuracy = 0
cmeans_accuracy2 = 0

for i, y in enumerate(Y):
    x = X[i]
    y_predicted1 = fcm.predict(x) + 1
    fcm_accuracy += y_predicted1 == y

    y_predicted2 = kmeans.predict([x])[0] + 1
    kmeans_accuracy += y_predicted2 == y

    y_predicted3 = cmeans_predict(np.expand_dims(x, 0).transpose(), cntr, m, error, maxiter=2)[0]
    y_predicted3 = np.argmax(y_predicted3, axis=0)[0] + 1
    cmeans_accuracy += y_predicted3 == y
    print(f"y: {y}, fcm: {y_predicted1} , kmeans: {y_predicted2}, cmeans:{y_predicted3}")

    # if np.argmax(y_predicted, axis=0) == y:
    #     print(f"equal {i}")
    # accuracy = accuracy + 1
    # else:
    #     print(y_predicted)

print(f"FCM Accuracy: {fcm_accuracy/len(X)}, K-Means Score: {kmeans_accuracy/len(X)}, CMeans: {cmeans_accuracy/len(X)}")
