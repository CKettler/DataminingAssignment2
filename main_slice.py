"""
main run file
"""

import data_aggregator as da
import data_slicer as sl

filepath = 'data/test_file1.csv'
data = da.DataAggregator(filepath)
sl.slicer(data.df, 10)