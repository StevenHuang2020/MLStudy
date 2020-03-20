#python3 steven
import numpy as np
import matplotlib.pyplot as plt

def getDataTxt(file,labelEndCol=True): #labelEndCol: label values at last/first column
    data = np.loadtxt(file, delimiter=',')
    print(data.shape)

    if labelEndCol:
        X = data[:, :-1]
        y = data[:, -1]
    else:
        X = data[:, 0]
        y = data[:, 1:]

    print('X.shape = ', X.shape)
    print('y.shape = ', y.shape)
    return X,y

def computeCost(X, y, theta): #loss mean square error
    h = np.dot(X, theta) - y
    J = np.dot(h, h) / (2 * X.shape[0])
    return J

def gradientDescent(X, y, theta, lr = 0.01, num_iters=10): #gradient descent training
    loss_ = np.zeros(num_iters)
    for i in range(num_iters):
        gradient = np.dot(X.T, (np.dot(X, theta) - y)) / X.shape[0]
        theta = theta - lr*gradient

        loss_[i] = computeCost(X, y, theta)

        if i % 100 == 0:
            print('epoch:',i,'gradient=',gradient,end='')
            for j in range(theta.shape[0]):
                print(' theta',j,'=',theta[j],',',end='')
            print('loss=',loss_[i])

    print(theta)
    return theta, loss_

def plotLoss(loss):
    plt.figure()
    plt.plot(loss)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    