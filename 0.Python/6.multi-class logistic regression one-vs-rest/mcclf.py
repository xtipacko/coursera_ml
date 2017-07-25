import numpy as np
from linclf import LinClassifier
# mc = multiclass
# sc = single class
class OnevsRestClf:
    def __init__(self, datapoints, a=0.1):
        y_column = datapoints.shape[1]-1
        y = np.rint(datapoints[:,y_column]).astype(int)
        classes = list(set(y))  
        # List of datapoints for one vs rest algroithm
        # contains whole datapoints for each class with 
        # binarized y, for class y=1, for other classes y = 0
        self.prepared_dp_lst = []
        self.class_order = []
        for c in classes:            
            new_dp = np.array([np.append(row[:y_column],1*int(int(row[y_column]) == c))
                                                                   for row in datapoints])      #splitting   
            self.class_order.append(c)
            self.prepared_dp_lst.append(new_dp)        
        #creating linear classifier for each generated dataset:
        self.clf_lst = []
        for prep_dp in self.prepared_dp_lst:
            clf = LinClassifier(prep_dp, a=a)
            self.clf_lst.append(clf)

    def predict(self, X):
        #based on max of predict LinClassifier
        all_predictions = []
        for clf in self.clf_lst:
            all_predictions.append(clf.predict(X))
        np_all_predictions = np.column_stack(all_predictions)
        classified_pred = []
        for row in np_all_predictions:
            prediction = None
            predict_class = None
            for c, i in enumerate(row):
                if prediction is None or prediction < i:
                    prediction = i
                    predict_class = self.class_order[c]
            classified_pred.append(predict_class)
        return np.array(classified_pred)


    def gradient_step(self):
        for clf in self.clf_lst:
            clf.gradient_step()


    def predictor_func(self, X):
        X = np.c_[np.ones(X.shape[0]),X]
        return self.predict(X)


    def CostFunction(self):
        cost = 0
        for clf in self.clf_lst:
            cost += clf.CostFunction()
        return cost