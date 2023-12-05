import numpy as np
from sklearn.cluster import KMeans

X = np.array([[5.9, 3.2],
              [4.6, 2.9],
              [6.2, 2.8],
              [4.7, 3.2],
              [5.5, 4.2],
              [5.0, 3.0],
              [4.9, 3.1],
              [6.7, 3.1],
              [5.1, 3.8],
              [6.0, 3.0]])

initial_centers = np.array([[6.2, 3.2],   
                            [6.6, 3.7],   
                            [6.5, 3.0]])  

kmeans = KMeans(n_clusters=3, init=initial_centers, random_state=0, n_init=1)
kmeans.fit(X)

c1 = kmeans.cluster_centers_[0]
print("c1 ai 1", np.round(c1, 3))

kmeans.fit(X)

c2 = kmeans.cluster_centers_[1]
print("c2 ai 2", np.round(c2, 3))



kmeans_convergence = KMeans(n_clusters=3, init=initial_centers, random_state=0)

kmeans_convergence.fit(X)

c3 = kmeans_convergence.cluster_centers_[2]
print("Cluster 3 center after convergence:", np.round(c3, 3))

print("Number of iterations for convergence:", kmeans_convergence.n_iter_)

plt.scatter(X[:, 0], X[:, 1], c=kmeans_convergence.labels_, cmap='viridis', alpha=0.5)
plt.scatter(kmeans_convergence.cluster_centers_[:, 0], kmeans_convergence.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()