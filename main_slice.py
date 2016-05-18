"""
main run file
"""

import data_aggregator as da
import data_slicer as sl

filepath = 'data/training_set_VU_DM_2014.csv'
data = da.DataAggregator(filepath)
sl.slicer(data.df, 24, max_num_slices=1)