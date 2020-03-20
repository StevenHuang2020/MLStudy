#python3 multiple variables linear regression
import numpy as np
import matplotlib.pyplot as plt
from common import *

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, ddof=1, axis=0)
    print('mu = ', mu)
    print('sigma = ', sigma)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def train(X,y):
    #print(X[:10])
    X_train, mu, sigma = featureNormalize(X)
    #print(X_train[:10])
    X_train = np.hstack((np.ones((X.shape[0], 1)), X_train))

    print('X_train = ',X_train[:3])
    print('y = ',y[:3])

    theta = np.zeros(X_train.shape[1]) #number of thetas,also equal nFeatures +1
    print('theta.shape=', theta.shape)
    lr = 0.01
    num_iters = 400
    theta, loss = gradientDescent(X_train, y, theta, lr, num_iters)
    return theta, loss  

def main():
    X,y = getDataTxt("./res/ex1data2.txt")
    theta, loss = train(X,y)
    #plotResult(X,y,theta)
    plotLoss(loss)

if __name__ == '__main__':
    main()
