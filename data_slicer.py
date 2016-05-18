import numpy as np
import random
import pandas as pd
import data_aggregator as da


def slicer(df, n_slices):
    k = 0
    rows = 0
    random_integers = []
    srch_ids = df['srch_id'].unique()
    print srch_ids
    while rows < 2000:
       rows = 2001




filepath = 'data/test_file1.csv'
data = da.DataAggregator(filepath)
slicer(data.df, 10)

