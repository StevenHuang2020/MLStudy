import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from common import *

def plotData(X,y):
    plt.figure()
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="y = 1")
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="y = 0")
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.legend()
    plt.show()

def costFunction(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    grad = np.dot(X.T, h - y) / X.shape[0]
    return J, grad

def cost(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    J = - (np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    return J

def gradient(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    grad = np.dot(X.T, h - y) / X.shape[0]
    return grad


def predict(theta, X):
    pass

def train(X,y): #Regularized logistic regression
    X_train = np.hstack((np.ones((X.shape[0], 1)), X))
    theta = np.array([0, 0, 0])
    pass


def main():
    X,y = getDataTxt('./res/ex2data2.txt')
    plotData(X,y)
    train(X,y)
    pass

if __name__ == '__main__':
    main()


