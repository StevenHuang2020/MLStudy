import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from Perceptron import *

def train(X,y):
    model = Perceptron(eta=0.001,n_iter=400)
    res = model.fit(X, y)

    print("w=", res.w_)
    #print("errors=", res.errors_)
    #showAfterTrainingErrors(res.errors_)
    return model

def prePareData():
    M = 100  #nSamples
    N = 4    #nFeatures
    df = pd.read_csv(r'..\..\dataBase\Iris\iris.data')
    y = df.iloc[0:M, -1].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:M, [0, 1]].values

    #print('X=',X)
    #print('y=',y)
    return X,y

def main():
    X,y = prePareData()
    model = train(X,y)

    #test four features
    #Value = [5.6, 2.9,3.8,1.5]
    #Value = [5.6, 3.8]
    #print(model.net_input(Value),model.predict(Value))
    
    ShowPrediction(X, y, model)

def showIrisData(df):
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='x', label='versicolor')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()
    pass

def showAfterTrainingErrors(errors_):
    plt.plot(range(1, len(errors_) + 1), errors_,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()

def ShowPrediction(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()