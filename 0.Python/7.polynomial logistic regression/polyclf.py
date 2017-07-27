import numpy as np

sigmoid = lambda x: np.reciprocal(1+np.exp(-1*x))

class PolyClassifier:

    def __init__(self, datapoints, a=0.1, degree=2):
        #t0, t1, t2, t1^2, t2^2, t1*t2
        self.theta   = np.zeros(6)     
        self.mu, self.std, self.X = self.normalize_data(datapoints[:,:-1], extract_stats=True, keep_lastcolumn=True)  
        self.X = self.polynomize_data(self.X, chop_lastcolumn=False, degree=2)
        self.y = datapoints[:,-1]
        self.a = a

 
    def normalize_data(self, datapoints, extract_stats=False, keep_lastcolumn=False):
        '''accepts prepared datapoints, with first column - all ones'''
        # mu0 = 0
        # std0 = 1
        if extract_stats:
            if keep_lastcolumn:
                mu_except_last = np.mean(datapoints[:,:-1], axis=0)
                mu = np.append(mu_except_last,np.zeros(1))
                std_except_last = np.std(datapoints[:,:-1], axis=0)
                std = np.append(std_except_last,np.ones(1))
            else:
                mu = np.mean(datapoints, axis=0)
                std = np.std(datapoints, axis=0)
        else:                        
            mu  = self.mu
            std = self.std

        datapoints = (datapoints - mu) / std

        return mu, std, datapoints


    def polynomize_data(self, datapoints, chop_lastcolumn=False, degree=2):
        dp_len = datapoints.shape[0]
        assert degree == 2, 'Feature Not Implemented yet'

        if chop_lastcolumn:
            orig_features   = datapoints[:,:-1]
        else:
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
        _, _, X = self.normalize_data(X)
        X = self.polynomize_data(X)
        return self.predict(X)

    def CostFunction(self):
        m = self.X.shape[0]
        h = self.predict(self.X) # hypothesis vector
        y = self.y
        y_t = y.transpose()
        return ( np.dot(-1*y_t,np.log(h)) - np.dot((1 - y).transpose(),np.log(1 - h)) ) / m
