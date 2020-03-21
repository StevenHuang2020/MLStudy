import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from common import *

def plotData(X,y,label1='Admitted',label2='Not admitted',f1='Exam 1 score',f2='Exam 2 score'):
    plt.figure()
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label=label1)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label=label2)
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.legend()
    plt.show()

def costFunction(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    grad = np.dot(X.T, h - y) / X.shape[0]
    return J, grad

def cost(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    #print('minh=', np.min(h))
    J = - (np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    return J

def gradient(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    grad = np.dot(X.T, h - y) / X.shape[0]
    return grad

def plotDecisionBoundary(theta, X, y,label1='Admitted',label2='Not admitted',f1='Exam 1 score',f2='Exam 2 score'):
    #x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    X_plot = np.hstack((np.ones((X_plot.shape[0], 1)), X_plot))
    y_plot = np.dot(X_plot, theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label=label1)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label=label2)
    plt.contour(xx, yy, y_plot, levels=[0])
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.legend()
    plt.show()

def predict(theta, X):
    X_train = np.hstack((np.ones((X.shape[0], 1)), X))
    prob = sigmoid(np.dot(X_train, theta))
    return (prob >= 0.5).astype(int)

def train(X,y): #logistic regression
    X_train = np.hstack((np.ones((X.shape[0], 1)), X))
    theta = np.array([0, 0, 0])
    J, grad = costFunction(theta, X_train, y)
    print(J)  # 0.693
    print(grad)

    res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y),
                        method='Newton-CG', jac=gradient)
    print(res.fun)  # 0.203
    print(res.x)  # -25.161 0.206 0.201
    plotDecisionBoundary(res.x, X, y)

    accuracy = np.mean(predict(res.x,X)==y)
    print('acc=',accuracy)
    pass

def getIrisData():
    '''To do a binary classifier use iris dataset 1~100 samples'''
    data = pd.read_csv('./res/iris.data')
    #print(data.head())

    #X = data.iloc[:100,:-1]
    X = data.iloc[:100,:2].to_numpy()
    y = data.iloc[:100,-1].to_numpy()

    y = np.where(y == 'Iris-setosa',0,1)
    #print(X[:5])
    #print(y[:5])
    #print(X[y == 1].iloc[:,0])
    #print(X[y == 0].iloc[:,1])
    print('X.shape = ', X.shape,type(X))
    print('y.shape = ', y.shape,type(y))
    return X,y

def main():
    #X,y = getDataTxt('./res/ex2data1.txt')
    #plotData(X,y)

    X,y = getIrisData()
    plotData(X,y,label1='Iris-versicolor',label2='Iris-setosa',f1='sepal length',f2='sepal weight')

    train(X,y)
    pass

if __name__ == '__main__':
    main()


