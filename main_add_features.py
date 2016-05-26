import data_preprocessing as dp
from datetime import datetime

filepath = 'data/data_slice_2.csv'

data_aggregator = dp.DataPreprocessing(filepath)

data_aggregator.add_data()

print 'data added', datetime.now().time()

data_aggregator.df.to_csv("data/data_slice_2_added_variables.csv")

print 'data saved', datetime.now().time()

filepath = 'data/data_slice_3.csv'

data_aggregator = dp.DataPreprocessing(filepath)

data_aggregator.add_data()

print 'data added', datetime.now().time()

data_aggregator.df.to_csv("data/data_slice_3_added_variables.csv")

print 'data saved', datetime.now().time()
