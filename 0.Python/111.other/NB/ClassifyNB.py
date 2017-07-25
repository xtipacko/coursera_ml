def classify(features_train, labels_train):   
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    #pred = clf.predict([1,1])
    return clf
    
    ### your code goes here!
    
    