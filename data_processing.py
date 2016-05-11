import pandas as pd

import numpy as np


class DataAggregator:
    def __init__(self, filepath, window_size=5, variables=None):
        self.df = pd.read_csv(filepath)
        self.participants = self.df['site_id'].unique()  # all participants (don't know if needed)
        self.variables = list(self.df.columns.values)  # all possible variable names
        print "data opened"

    def read(self, ):
        """
        Generates the aggregated data.
        There are a few possible variables.
        :param method: Method of the aggregation.
        Some means that only a few features are used.
        """

        if method == 'some':
            return self.window_some_features()
        elif method == 'cor':
            targets = [self.df['click_bool'], self.df['booking_bool']]
            return self.df, self.variables, targets

    method = 'cor'
    def all_data(self):
        data_frames_list = []
        targets = []
        for current_id in self.participants:
            mask = (self.df['site_id'] == current_id)

            # get every date for one person
            tf = self.df.loc[mask]
            some_feature_data = pd.concat([tf['site_id'], tf['visitor_location_country_id'], tf['prop_country_id'],
                                           tf['prop_starrating'], tf['prop_brand_bool'], tf['price_usd']], axis=1,
                                          keys=['id', 'location visitor', 'location prop', 'star rating', 'brand boolean', 'price usd'])
            target = tf['click_bool']
            data_frames_list.append(some_feature_data)
            targets.append(target)

        return data_frames_list, targets

    def window_some_features(self):
        data_frames_list, targets = self.all_data()
        return data_frames_list, targets

