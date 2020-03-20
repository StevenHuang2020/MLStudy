#python3 single variable linear regression
import numpy as np
import matplotlib.pyplot as plt
from common import computeCost,gradientDescent

def getData():
    data = np.loadtxt("./res/ex1data1.txt", delimiter=',')

    X = data[:, 0][:, np.newaxis]
    print(X.shape)
    #X = data[:, -1]
    #print(X.shape)
    y = data[:, -1]
    print(y.shape)
    return X,y

def plotXY(X,y):
    plt.figure()
    plt.scatter(X[:,0], y)
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show()

def train(X,y):
    X_train = np.hstack((np.ones((X.shape[0], 1)), X))
    theta = np.zeros(2)
    lr = 0.01
    num_iters = 500
    theta, loss = gradientDescent(X_train, y, theta, lr, num_iters)
    return theta, loss 

def plotResultAndLoss(X,y,theta,loss):
    ax = plt.subplot(1,2,1)
    ax.scatter(X,y)
    ax.set_title('Train dataset')

    X_plot = np.linspace(5, 23, 100)
    ax.plot(X_plot, theta[0] + X_plot * theta[1], color="r", linewidth=2)
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")

    ax = plt.subplot(1,2,2)
    ax.plot(loss)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')

    #fig.suptitle("Linear regresison grident descent", fontsize=14)
    plt.show()

def plotResult(X,y,theta):
    plt.figure()
    plt.scatter(X, y)
    X_plot = np.linspace(5, 23, 100)
    plt.plot(X_plot, theta[0] + X_plot * theta[1], color="r", linewidth=2)
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.xlim(5, 23)
    plt.show()

def main():
    X,y = getData()
    #plotXY(X,y)
    theta, loss = train(X,y)
    plotResultAndLoss(X,y,theta,loss)

if __name__ == '__main__':
    main()
