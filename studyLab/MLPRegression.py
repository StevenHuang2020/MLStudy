"""
Steven 20/03/2020
MLPRegressor solve regression problems
"""
#python3 steven
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from plotCommon import *

def noise():
    return np.random.rand(1)

def fuc(x):
    #return 2*x + 3 + noise()/10
    return x**2 + 3*x + 5 + noise()*0.8

def prepareData(N=10):
    X = np.linspace(1, 10, N).reshape((N, 1))
    y = fuc(X)
    #print(X)
    #print(y)
    return X,y

def createModel():
    #return LinearRegression()
    #return MLPRegressor(hidden_layer_sizes = (100,10,10), activation = 'relu', solver = 'sgd',max_iter=500,warm_start=True)
    return MLPRegressor(hidden_layer_sizes = (10,10), max_iter=100,warm_start=True)
    #return SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    #return SVR(kernel='linear', C=100, gamma='auto')
    #return SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1)
    #return KNeighborsRegressor(n_neighbors=5)
    #return DecisionTreeRegressor()
    #return RandomForestRegressor()

def trainWarmStart(x_train, y_train):
    reg = createModel()
    iter = 200
    for i in range(iter):
        reg.fit(x_train, np.ravel(y_train, order='C'))
        lossTrain = mean_squared_error(reg.predict(x_train),y_train)
        if i % 50 == 0:
            print(i,lossTrain,reg.loss_)
            #print(i,lossTrain)
    print('scores = ',reg.score(x_train,y_train))
    return reg

def plotSubplot(x_train, y_train, x, y, loss=None):
    plt.subplot(1, 2, 1)
    plt.title('Training dataset')

    plt.scatter(x_train, y_train, label='input dataset')  # plot train dataset
    plt.plot(x, y, color='r',label='predict') #pred dataset
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Training Loss')
    if loss:
        plt.plot(loss, color='g')
    plt.show()

def main():
    X,y = prepareData(20)
    reg = trainWarmStart(X,y)
    #plotLoss(reg.loss_curve_)

    test = np.arange(1,12)
    test_tar = fuc(test)
    test =test.reshape((len(test),1))

    pred = reg.predict(test)
    print(test.T)
    print(test_tar)
    print(pred.T)
    plotSubplot(X,y,test,pred,reg.loss_curve_)


if __name__=='__main__':
    main()

