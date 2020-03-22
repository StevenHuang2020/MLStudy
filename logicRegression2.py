import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from common import getDataTxt, sigmoid
from logicRegression import gradient

def plotData(X,y):
    plt.figure()
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="y = 1")
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="y = 0")
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.legend()
    plt.show()

def predict(theta, X):
    pass

def mapFeature(X1, X2):
    degree = 6
    out = np.ones((X1.shape[0], 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.hstack((out, (X1 ** (i - j) * X2 ** j)[:, np.newaxis]))
    return out

def cost(theta, X, y, lam):
    h = sigmoid(np.dot(X, theta))
    t = np.zeros(len(theta))
    t[1:] = theta[1:]
    J = (-(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
         + lam * np.dot(t, t) / (2 * X.shape[0]))
    return J

def gradient(theta, X, y, lam):
    h = sigmoid(np.dot(X, theta))
    t = np.zeros(len(theta))
    t[1:] = theta[1:]
    grad = np.dot(X.T, h - y) / X.shape[0] + lam * t / X.shape[0]
    return grad

def costFunctionReg(theta, X, y, lam):
    h = sigmoid(np.dot(X, theta))
    t = np.zeros(len(theta))
    t[1:] = theta[1:]
    J = (-(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
         + lam * np.dot(t, t) / (2 * X.shape[0]))
    grad = np.dot(X.T, h - y) / X.shape[0] + lam * t / X.shape[0]
    return J, grad

def predict(theta, X):
    X_train = mapFeature(X[:, 0], X[:, 1])
    prob = sigmoid(np.dot(X_train, theta))
    return (prob >= 0.5).astype(int)

def plotDecisionBoundary1(theta, X, y,ax=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    X_plot = mapFeature(X_plot[:, 0], X_plot[:, 1])
    y_plot = np.dot(X_plot, theta).reshape(xx.shape)

    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="y = 1")
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="y = 0")
    ax.contour(xx, yy, y_plot, levels=[0])
    ax.set_xlabel("Microchip Test 1")
    ax.set_ylabel("Microchip Test 2")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.legend()


def plotDecisionBoundary(theta, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    X_plot = mapFeature(X_plot[:, 0], X_plot[:, 1])
    y_plot = np.dot(X_plot, theta).reshape(xx.shape)

    plt.figure()
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="y = 1")
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="y = 0")
    plt.contour(xx, yy, y_plot, levels=[0])
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.show()

def main():
    X,y = getDataTxt('./res/ex2data2.txt')
    plotData(X, y)
    X_train = mapFeature(X[:, 0], X[:, 1])
    print('X_train.shape=', X_train.shape)
    #print(X_train[:5])

    if 0:
        lam = 1
        theta = np.zeros(X_train.shape[1])
        print('theta.shape=', theta.shape)
        J, grad = costFunctionReg(theta, X_train, y, lam)
        print(J)  # 0.693
        print('grad.shape = ', grad.shape)
        print(grad[:5])

    lamList=[1,0,100] #normal, overfitting, underfitting
    if 0:
        lam = 100 #1 0
        theta = np.zeros(X_train.shape[1])
        res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y, lam),
                                method='Newton-CG', jac=gradient)
        print(res.fun)
        print(res.x[:5])

        acc = np.mean(predict(res.x, X) == y)
        print('acc = ',acc)
        plotDecisionBoundary(res.x, X, y)
    else:
        for i,lam in enumerate(lamList):
            theta = np.zeros(X_train.shape[1])
            res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y, lam),
                                    method='Newton-CG', jac=gradient)

            ax = plt.subplot(2, 2, i+1)
            plotDecisionBoundary1(res.x, X, y,ax)
        plt.show()
    #train(X, y)
    pass

def train(X,y): #Regularized logistic regression
    X_train = np.hstack((np.ones((X.shape[0], 1)), X))
    theta = np.array([0, 0, 0])
    pass

if __name__ == '__main__':
    main()


