import numpy as np

sigmoid = lambda x: np.reciprocal(1+np.exp(-1*x))

class PolyClassifier:

    def __init__(self, datapoints, a=0.1, degree=2):
        #t0, t1, t2, t1^2, t2^2, t1*t2
        self.theta   = np.zeros(6)     
        self.X = self.polynomize_data(datapoints[:,:-1], degree=2)
        self.mu, self.rng, self.X = self.normalize_data(self.X, extract_stats=True)
        self.y = datapoints[:,-1]
        self.a = a

 
    def normalize_data(self, datapoints, extract_stats=False):
        '''accepts prepared datapoints, with first column - all ones'''
        # mu0 = 0
        # std0 = 1
        if extract_stats:
            mu_except_first = np.mean(datapoints[:,1:], axis=0)
            mu = np.append(np.zeros(1),mu_except_first)
            max_except_first = np.max(datapoints[:,1:], axis=0)
            min_except_first = np.min(datapoints[:,1:], axis=0)
            rng = np.append(np.ones(1),max_except_first-min_except_first)
        else:                        
            mu  = self.mu
            rng = self.rng

        datapoints = (datapoints - mu) / rng

        return mu, rng, datapoints


    def polynomize_data(self, datapoints, degree=2):
        dp_len = datapoints.shape[0]
        assert degree == 2, 'Feature Not Implemented yet'

        orig_features = datapoints
        squred_features = orig_features**2
        feature_combinations = np.multiply(datapoints[:,0],datapoints[:,1])
        return np.column_stack((np.ones(dp_len),
                                orig_features,
                                squred_features,
                                feature_combinations))


    def gradient_step(self):
        self.theta = self.theta - self.a*self.grad()


    def predict(self, X):
        return sigmoid(X.dot(self.theta))


    def grad(self):
        m = self.X.shape[0]
        h = self.predict(self.X) # hypothesis vector                  
        return (self.X.transpose().dot((h - self.y))) / m

    def predictor_func(self, X):        
        X = self.polynomize_data(X)
        _, _, X = self.normalize_data(X)
        return self.predict(X)

    def CostFunction(self):
        m = self.X.shape[0]
        h = self.predict(self.X) # hypothesis vector
        y = self.y
        y_t = y.transpose()
        return ( np.dot(-1*y_t,np.log(h)) - np.dot((1 - y).transpose(),np.log(1 - h)) ) / m
