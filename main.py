"""
main run file
"""

import data_processing as dr
import numpy as np
import csv
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

filepath = 'data/test_file1.csv'

data_aggregator = dr.DataAggregator(filepath)

data_frames, targets = data_aggregator.read()