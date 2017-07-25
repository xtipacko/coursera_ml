import numpy as np

sigmoid = lambda x: np.reciprocal(1+np.exp(-1*x))

class LinClassifier:

    def __init__(self, datapoints, a=0.1):
        self.theta   = np.zeros(3)
        self.rows    = datapoints.shape[0]
        self.columns = datapoints.shape[1]
        self.X = np.c_[np.ones(self.rows),datapoints[:,:self.columns-1]]
        self.y = datapoints[:,self.columns-1]
        self.a = a


    def gradient_step(self):
        self.theta = self.theta - self.a*self.grad()


    def predict(self, X):
        return sigmoid(X.dot(self.theta))


    def grad(self):
        m = self.X.shape[0]
        h = self.predict(self.X) # hypothesis vector          
        return (self.X.transpose().dot((h - self.y))) / m


    def CostFunction(self):
        m = self.X.shape[0]
        h = self.predict(self.X) # hypothesis vector
        y = self.y
        y_t = y.transpose()
        return ( np.dot(-1*y_t,np.log(h)) - np.dot((1 - y).transpose(),np.log(1 - h)) ) / m
