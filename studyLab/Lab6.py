from IPython.display import Image
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

def getData():
    X, y = make_blobs(n_samples=150, 
                    n_features=2, 
                    centers=3, 
                    cluster_std=0.5, 
                    shuffle=True, 
                    random_state=0)
    print('X.shape=',X.shape)
    print('y.shape=',y.shape)
    return X,y

def plotDataSet(X):
    plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolor='black', s=50)
    plt.grid()
    plt.tight_layout()
    #plt.savefig('C:/Users/rpear/Desktop/11_01.png', dpi=300)
    plt.show()

def KMeansRandom(X):
    km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    y_km = km.fit_predict(X)
    
    print('y_km=',y_km)
    print('cluster_centers_=', km.cluster_centers_)
    print('Distortion: %.2f' % km.inertia_) #Sum of squared distances of samples to their closest cluster center
    
    plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
                s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1')
    plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
                s=50, c='orange', marker='o', edgecolor='black', label='cluster 2')
    plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1],
                s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                s=250, marker='*', c='red', edgecolor='black',  label='centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    #plt.savefig('C:/Users/rpear/Desktop/11_02.png', dpi=300)
    plt.show()

def KMeansSearch(X):
    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)
    print('distortions=',distortions)
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    #plt.savefig('C:/Users/rpear/Desktop/11_03.png', dpi=300)
    plt.show()

def KMeansSilhouette(X):
    # ## Quantifying the quality of clustering  via silhouette plots
    km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    y_km = km.fit_predict(X)

    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    print('cluster_labels,n_clusters=',cluster_labels,n_clusters)
    return plotSilhouetteValues(X,y_km)

def KMeansBad(X):
    # Comparison to "bad" clustering:
    km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300,tol=1e-04, random_state=0)
    y_km = km.fit_predict(X)

    print('y_km=',y_km)
    print('cluster_centers_=', km.cluster_centers_)
    print('Distortion: %.2f' % km.inertia_)
    
    plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
                s=50, c='lightgreen', edgecolor='black', marker='s', label='cluster 1')
    plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
                s=50, c='orange', edgecolor='black', marker='o', label='cluster 2')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                s=250, marker='*', c='red', label='centroids')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    #plt.savefig('C:/Users/rpear/Desktop/11_05.png', dpi=300)
    plt.show()
    return y_km

def plotSilhouetteValues(X, y_km):
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
                edgecolor='none', color=color)

        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)
        
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--") 

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    plt.tight_layout()
    #plt.savefig('C:/Users/rpear/Desktop/11_06.png', dpi=300)
    plt.show()

def main():
    X,y = getData()
    plotDataSet(X)
    KMeansRandom(X)
    KMeansSearch(X)
    KMeansSilhouette(X)
    
    sv = KMeansBad(X)
    plotSilhouetteValues(X,sv)
    
if __name__=='__main__':
    main()
    