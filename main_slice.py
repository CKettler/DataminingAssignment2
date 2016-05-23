"""
main run file
"""

import data_aggregator as da
import data_slicer as sl

filepath = 'data/training_set_VU_DM_2014.csv'
data = da.DataAggregator(filepath)
data.read_data()
sl.slicer(data.df, 24, start_slice=3, outputfiles='data/data_slice_%d.csv')
print "done"