import data_aggregator as da
import numpy as np
from sklearn import dummy
from datetime import datetime

def classification(self, data_matrix, target_matrix, test_matrix, strategy = 'most_frequent'):
    print "data detected", datetime.now().time()
    model = dummy.DummyClassifier(strategy=strategy, random_state=None, constant=None)
    print "model made", datetime.now().time()
    model.fit(data_matrix, target_matrix)
    print "model fitted", datetime.now().time()
    results = model.predict(test_matrix)
    print results




filepath = 'data/data_slice_1_added_variables.csv'

data_aggregator = da.DataAggregator(filepath)
df = data_aggregator.read_data()
variables = list(df.columns.values)
print variables
target = data_aggregator.df['target'].as_matrix()
print target
# df = data_aggregator.keep_df_variables()
#
#
# dummy_classifier = dc.(df, targets, test_df)