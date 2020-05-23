# imoorting libraires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# loading dataset
dataset = pd.read_csv('Python/Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow methods')
plt.xlabel("Number of clusters")
plt.ylabel("WSS")
plt.show()

# training the KMeans model on dataset
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kemans = kmeans.fit_predict(X)

plt.scatter(X[y_kemans == 0, 0], X[y_kemans == 0, 1], s=100, c='red',
            label='Cluster 1')
plt.scatter(X[y_kemans == 1, 0], X[y_kemans == 1, 1], s=100, c='blue',
            label='Cluster 2')
plt.scatter(X[y_kemans == 2, 0], X[y_kemans == 2, 1], s=100, c='green',
            label='Cluster 3')
plt.scatter(X[y_kemans == 3, 0], X[y_kemans == 3, 1], s=100, c='cyan',
            label='Cluster 4')
plt.scatter(X[y_kemans == 4, 0], X[y_kemans == 4, 1], s=100, c='magenta',
            label='Cluster 5')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
