import numpy as np
#Implementing a perceptron learning algorithm in Python

class Perceptron(object):
    """Perceptron classifier.
    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    errors_ : list
    Number of misclassifications in every epoch.
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        #print('init called...')

    def fit(self, X, y):
        """Fitting training data.
        Parameters
        --------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : object
        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            if 0:
                for xi, target in  zip(X, y):
                    print(xi,target)
            elif 1:
                for xi, target in  zip(X, y):
                    update = self.eta * (target - self.predict(xi))
                    #print('update=',update)

                    self.w_[1:] += update * xi  #derivative
                    self.w_[0] += update
                    errors += int(update != 0.0)
                self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
        
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    def print(self):
        #print("w=", self.w_)
        #print("errors=", self.errors_)
        pass

def trainingDataSetAA():
    #unit step function
    #X = np.random.rand(5).reshape((5, 1))
    #y = np.where(X > 0.5, 1, -1)
    
    """linear dataset,one pararmeter"""
    #X = np.array([1, 2, 3, 4, 5]).reshape((5,1))
    #y = np.where(X > 3.5, 1, -1)

    """two features dataset,two pararmeter"""
    n_samples = 1000
    n_features = 5
    X = np.random.rand(n_samples*n_features).reshape((n_samples,n_features))
    y = np.zeros(n_samples).reshape((n_samples,1))

    for i in range(n_samples):
        #print(X[i],X[i][0],X[i][1])
        y[i] = np.where(X[i][0] > X[i][1], 1, -1) #lable only two class
   
    print(X)
    print(y)
    return X, y

def trainingDataSet(nSamples=10, nFeatures=2): #samples and features numbers
    #unit step function
    #X = np.random.rand(5).reshape((5, 1))
    #y = np.where(X > 0.5, 1, -1)
    
    """linear dataset,one pararmeter"""
    #X = np.array([1, 2, 3, 4, 5]).reshape((5,1))
    #y = np.where(X > 3.5, 1, -1)

    """dataset,n_samples,n_features pararmeter"""
    n_samples = nSamples
    n_features = nFeatures
    X = np.random.rand(n_samples*n_features).reshape((n_samples,n_features))
    y = np.zeros(n_samples).reshape((n_samples,1))

    for i in range(n_samples):
        #print(X[i],X[i][0],X[i][1])
        y[i] = np.where(X[i][0] > X[i][1], 1, -1)
   
    #print(X)
    #print(y)
    return X, y

def calculateModelAccuracy(model,nSamples,nFeatures):
    test_X, test_y = trainingDataSet(nSamples,nFeatures)

    predict = model.predict(test_X)
    test_y = test_y.transpose().flatten()
    #print(test_y)
    #print(predict)

    correct = np.where(test_y == predict, 1, 0)
    accuracy = np.sum(correct) * 100.0 / len(correct) 
    #print(accuracy)
    
    acc = np.mean(test_y == predict)
    print("acc=",acc)
    
    print('predict accuracy is: {}%'.format(accuracy))

def main():
    N = 1000
    M = 2

    X, y = trainingDataSet(N,M)
    model = Perceptron()
    res = model.fit(X, y)

    print("w=", res.w_)
    print("errors=", res.errors_)

    #test one features
    #print(np.dot(X, res.w_[1:]) + res.w_[0])
    #Value = 3.0
    #print(model.net_input(Value),model.predict(Value))

    #test  features must qual to training dataset features
    #Value = [3.0, 5.8, 1.0, 2.0, 1.6]#Value = [3.0, 5.8]
    #print(model.net_input(Value),model.predict(Value))

    calculateModelAccuracy(model,nSamples=N,nFeatures=M)
    
if __name__ == '__main__':
    main()
