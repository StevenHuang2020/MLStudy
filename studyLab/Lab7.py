from sklearn.datasets import make_blobs
import matplotlib.pyplot as pltp
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

from Lab6 import getData,plotDataSet

def getDataMoons():
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    print('X.shape=',X.shape)
    print('y.shape=',y.shape)
    return X,y

def plotData(X):
    plt.scatter(X[:, 0], X[:, 1])
    plt.tight_layout()
    plt.show()

def KMeansModel(X,N=3):
    km = KMeans(n_clusters=N, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    y_km = km.fit_predict(X)
    print('y_km=',y_km)
    return y_km

def AgglomerativeModel(X,N=3):
    ac = AgglomerativeClustering(n_clusters=N, affinity='euclidean', linkage='complete')
    labels = ac.fit_predict(X)
    print('Cluster labels: %s' % labels)
    return labels

def plotCompare(X, y_km, y_ac):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.set_title('K-means clustering')
    ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
            edgecolor='black', c='lightblue', marker='o', s=40, label='cluster 1')
    ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
                edgecolor='black', c='red', marker='s', s=40, label='cluster 2')
    
    ax2.set_title('Agglomerative Clustering')
    ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c='lightblue',
            edgecolor='black', marker='o', s=40, label='cluster 1')
    ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c='red',
                edgecolor='black', marker='s', s=40, label='cluster 2')
    plt.tight_layout()
    plt.show()


# Now apply the DBSCAN clusterer with the lower bound value of min_samples 
# discussed in the lecture. The eps parameter is harder to set; in this dataset 
# the default value of 0.5 did not discover two clusters

def DBSCANClustering(X):
    db = DBSCAN(eps=0.3, min_samples=4, metric='euclidean')
    y_db = db.fit_predict(X)
    plt.scatter(X[y_db == 0, 0], X[y_db == 0, 1],
                c='lightblue', marker='o', s=40, edgecolor='black', label='cluster 1')
    plt.scatter(X[y_db == 1, 0], X[y_db == 1, 1],
                c='red', marker='s', s=40, edgecolor='black', label='cluster 2')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    X,y = getData()
    KMeansModel(X)
    AgglomerativeModel(X,3)
    AgglomerativeModel(X,2)
    
    X,y = getDataMoons()
    plotDataSet(X)
    #plotData(X)
    plotCompare(X, KMeansModel(X,2), AgglomerativeModel(X,2))
    DBSCANClustering(X)
    
if __name__=='__main__':
    main()