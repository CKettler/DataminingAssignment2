"""
main run file
"""

import data_preprocessing as dp
from datetime import datetime
import numpy as np
import csv
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

#filepath = 'data/training_set_VU_DM_2014.csv'
filepath = 'data/data_slice_1.csv'
#filepath = 'data/mega_small.csv'


data_aggregator = dp.DataPreprocessing(filepath)

data_aggregator.add_data()

print 'data added', datetime.now().time()

data_aggregator.df.to_csv("data/data_slice_1_added_variables.csv")

print 'data saved', datetime.now().time()