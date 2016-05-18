"""
main run file
"""

import data_aggregator as da
import data_slicer as sl

filepath = 'data/data_slice_1.csv'
data = da.DataAggregator(filepath)
sl.slicer(data.df, 24, outputfiles='data/small_data_slice_%d.csv', max_num_slices=1)
print "done"