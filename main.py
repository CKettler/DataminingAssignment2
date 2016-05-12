"""
main run file
"""

import data_processing as dr
from datetime import datetime
import numpy as np
import csv
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

filepath = 'data/training_set_VU_DM_2014.csv'

data_aggregator = dr.DataAggregator(filepath)

data_aggregator.add_data()

print 'data added', datetime.now().time()

data_aggregator.df.to_csv("data/added_variables.csv")

print 'data saved', datetime.now().time()