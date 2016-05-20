import numpy as np
from sklearn import dummy
from datetime import datetime

class DummyClassifier:
    def __init__(self, data_matrix, target, test_matrix,  variables=None):
        self.data_matrix = data_matrix
        self.target = target
        self.test_matrix = test_matrix
        print "data detected", datetime.now().time()

    def classification(self, strategy = 'most_frequent'):
        model = dummy.DummyClassifier(strategy=strategy, random_state=None, constant=None)
        model.fit(self.data_matrix,self.targets_matrix)
        results = model.predict(self.test_matrix)
        print results


    