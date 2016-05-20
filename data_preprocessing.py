import pandas as pd
from datetime import datetime
from data_aggregator import *
import numpy as np
import math


class DataPreprocessing(DataAggregator):
    def __init__(self, filepath, variables=None):
        self.df = pd.DataFrame()
        self.filepath = filepath
        self.read_data()

        print "data opened", datetime.now().time()

    def read(self, method = 'some'):
        """
        Generates the aggregated data.
        There are a few possible variables.
        :param method: Method of the aggregation.
        Some means that only a few features are used.
        """

        if method == 'some':
            return self.window_some_features()
        elif method == 'cor':
            targets = [self.df['click_bool'], self.df['booking_bool'], self.df['target']]
            return self.df, self.variables, targets


    def add_data(self):

        lambdafunc = lambda x: pd.Series([x['date_time'].hour / 6,
                                         self.comp_rate_calc([x['comp%d_rate' % i] for i in range(1, 9)]),
                                         self.comp_expensive_calc([x['comp%d_rate' % i] for i in range(1, 9)]),
                                         self.comp_cheap_calc([x['comp%d_rate' % i] for i in range(1, 9)]),
                                         int(x['click_bool']) + 5 * int(x['booking_bool'])
                                         ])

        print "lambdafunc created", datetime.now().time()
        newcols = self.df.apply(lambdafunc, axis=1)
        print "new cols created", datetime.now().time()
        newcols.columns = ['time_class', 'comp_rate_sum', 'comp_expensive', 'comp_cheap', 'target']
        booking_properties_dict =  self.bookings_property()
        booking_prop_array, found_prop_array = self.feature_from_booking_properties(booking_properties_dict)
        self.df['no_bookings_prop'] = np.transpose(booking_prop_array)
        self.df['no_found_prop'] = np.transpose(found_prop_array)
        self.df = self.df.join(newcols)
        print "new df joined", datetime.now().time()

    def comp_rate_calc(self, comps_list):
        result = np.sum(comps_list)
        return result if not math.isnan(result) else 0.0

    def comp_expensive_calc(self, comps_list):
        min = np.min(comps_list)
        return 1 if min < 0 else 0

    def comp_cheap_calc(self, comps_list):
        max = np.max(comps_list)
        return 1 if max > 0 else 0


    def bookings_property(self):
        # creates dictionary of hotels, returning 123(number hotel): 3(number counts)
        booking_properties_dict = {}
        rows, cols = self.df.shape
        for i in range(0,rows):
            if self.df['booking_bool'][i] == 1:
                property_id = self.df['prop_id'][i]
                if str(property_id) in booking_properties_dict:
                    booking_properties_dict[str(property_id)] += 1
                else:
                    booking_properties_dict[str(property_id)] = 1

        return booking_properties_dict


    def feature_from_booking_properties(self, booking_properties_dict):
        rows, cols = self.df.shape
        feature_array_bookings = np.zeros(rows)
        feature_array_found = np.zeros(rows)
        for key in booking_properties_dict:
            indexes_prop = self.df[self.df['prop_id'] == int(key)].index.tolist()
            # number of times a property was found in the data = no_times_found
            no_times_found = len(indexes_prop)
            for index in indexes_prop:
                feature_array_bookings[index] = booking_properties_dict[key]
                feature_array_found[index] = no_times_found
        return feature_array_bookings, feature_array_found



