import numpy as np
from itertools import combinations_with_replacement
from functools import reduce

sigmoid = lambda x: np.reciprocal(1+np.exp(-1*x))

class PolyClassifier:

    def __init__(self, datapoints, a=0.1, l=0.1, degree=2, init_norm=False):
        self.degree = degree
        self.init_norm = init_norm
        #t0, t1, t2, t1^2, t2^2, t1*t2
        self.dp_len = datapoints.shape[0]
        self.dp_feat = datapoints.shape[1] - 1
        feat_num = lambda d: 1 + 2*d + (d**2 - d) // 2 # for 2 initial features
        self.theta   = np.zeros(feat_num(degree))
        self.imu, self.istd = None, None
        self.X = self.normalize_initial_data(datapoints[:,:-1], extract_stats=True)        
        self.X = self.polynomize_data(self.X)
        self.mu, self.rng, self.X = self.normalize_data(self.X, extract_stats=True)        
        self.y = datapoints[:,-1]
        self.a = a
        self.l = l


    def normalize_initial_data(self, datapoints, extract_stats=False):
        if self.init_norm:
            if extract_stats:
                self.imu  = np.mean(datapoints, axis=0)
                self.istd = np.std(datapoints, axis=0)
            return (datapoints - self.imu) / self.istd
        else:
            return datapoints

 
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


    def polynomize_data(self, datapoints):
        dp_len = datapoints.shape[0]

        x0 = np.ones(dp_len)
        orig_features = list(datapoints.transpose())

        # for combinations could have used binary counting, with each bit representing element in a list
        all_comb_terms = []
        for i in range(2, self.degree+1):
            combinations = combinations_with_replacement(orig_features,i)
            all_comb_terms.extend(combinations)
        all_features = [x0] + orig_features
        for term in all_comb_terms:
            multple = reduce(np.multiply,term)
            all_features.append(multple)

        return np.column_stack(all_features)        


    def gradient_step(self):
        self.theta = self.theta - self.a*self.grad()


    def predict(self, X):
        return sigmoid(X.dot(self.theta))


    def grad(self):
        m = self.X.shape[0]
        h = self.predict(self.X) # hypothesis vector   
        theta_except0 =  np.append([0],self.theta[1:])
        reg_term = self.l*theta_except0 #regiularization term
        return (self.X.transpose().dot((h - self.y)) + reg_term) / m 


    def predictor_func(self, X):    
        X = self.normalize_initial_data(X)  
        X = self.polynomize_data(X)
        _, _, X = self.normalize_data(X)
        return self.predict(X)


    def CostFunction(self):
        m = self.X.shape[0]
        h = self.predict(self.X) # hypothesis vector
        y = self.y
        y_t = y.transpose()
        theta_1_n = self.theta[1:]
        l = self.l
        reg_term_for_cost = l*theta_1_n.dot(theta_1_n) / (2*m)
        return ( np.dot(-1*y_t,np.log(h)) - np.dot((1 - y).transpose(),np.log(1 - h)) ) / m + reg_term_for_cost
