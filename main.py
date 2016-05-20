"""
main run file
"""

import data_preprocessing as dp
import data_aggregator as da
from datetime import datetime
import dummy_classifier as dc
import numpy as np
import csv
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

#filepath = 'data/training_set_VU_DM_2014.csv'
filepath = 'data/data_slice_1.csv'
#filepath = 'data/mega_small.csv'


#data_aggregator = dp.DataPreprocessing(filepath)

#data_aggregator.add_data()

#print 'data added', datetime.now().time()

#data_aggregator.df.to_csv("data/data_slice_1_added_variables.csv")

#print 'data saved', datetime.now().time()

filepath2 = 'data/data_slice_1_added_variables.csv'

data_aggregator = da.DataAggregator(filepath2)
df = data_aggregator.read_data()
print variables = list(df.columns.values)
target = data_aggregator.df['target'].as_matrix()
print target
# df = data_aggregator.keep_df_variables()
#
#
# dummy_classifier = dc.(df, targets, test_df)