import pandas as pd
from datetime import datetime
import numpy as np
import math


class DataAggregator:
    def __init__(self, filepath, window_size=5, variables=None):
        print "start the process", datetime.now().time()
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        self.df = pd.read_csv(filepath, parse_dates=['date_time'], date_parser=dateparse)
        self.df[['comp%d_rate' % i for i in range(1, 9)]] = self.df[['comp%d_rate' % i for i in range(1, 9)]].fillna(axis=1, method='backfill')
        self.site_id = self.df['site_id'].unique()  # all participants (don't know if needed)
        self.variables = list(self.df.columns.values)  # all possible variable names
        print self.variables

        print "data opened", datetime.now().time()

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
        for current_id in self.site_id:
            mask = (self.df['site_id'] == current_id)

            # get every date for one person
            tf = self.df.loc[mask]
            # time_class =datetime.strptime(tf['date_time'].get(0), '%Y-%m-%d %H:%M:%S').hour/6
            tf['time_class'] = tf.apply(lambda x: x['date_time'].hour/6, axis=1)
            tf['comp_rate_sum'] = tf.apply(lambda x: self.comp_rate_calc([x['comp%d_rate' % i] for i in range(1, 9)]), axis=1)
            tf['comp_expensive'] = tf.apply(lambda x: self.comp_expensive_calc([x['comp%d_rate' % i] for i in range(1, 9)]), axis=1)
            tf['comp_cheap'] = tf.apply(lambda x: self.comp_cheap_calc([x['comp%d_rate' % i] for i in range(1, 9)]), axis=1)
            tf['target'] = tf.apply(lambda x: int(x['click_bool'])+4*int(x['booking_bool']), axis=1)
            print tf.loc[:, ('time_class', 'comp_rate_sum', 'comp_expensive', 'comp_cheap', 'target')]
            some_feature_data = pd.concat([tf['site_id'], tf['visitor_location_country_id'], tf['prop_country_id'],
                                           tf['prop_starrating'], tf['prop_brand_bool'], tf['price_usd']], axis=1,
                                          keys=['id', 'location visitor', 'location prop', 'star rating', 'brand boolean', 'price usd'])
            target = tf['click_bool']
            data_frames_list.append(some_feature_data)
            targets.append(target)

        return data_frames_list, targets

    def add_data(self):
        # self.df['time_class'] = self.df.apply(lambda x: x['date_time'].hour / 6, axis=1)
        # self.df['comp_rate_sum'] = self.df.apply(lambda x: self.comp_rate_calc([x['comp%d_rate' % i] for i in range(1, 9)]),
        #                                axis=1)
        # self.df['comp_expensive'] = self.df.apply(lambda x: self.comp_expensive_calc([x['comp%d_rate' % i] for i in range(1, 9)]),
        #                                 axis=1)
        # self.df['comp_cheap'] = self.df.apply(lambda x: self.comp_cheap_calc([x['comp%d_rate' % i] for i in range(1, 9)]), axis=1)
        # self.df['target'] = self.df.apply(lambda x: int(x['click_bool']) + 4 * int(x['booking_bool']), axis=1)

        lambdafunc = lambda x: pd.Series([x['date_time'].hour / 6,
                                          self.comp_rate_calc([x['comp%d_rate' % i] for i in range(1, 9)]),
                                          self.comp_expensive_calc([x['comp%d_rate' % i] for i in range(1, 9)]),
                                          self.comp_cheap_calc([x['comp%d_rate' % i] for i in range(1, 9)]),
                                          int(x['click_bool']) + 4 * int(x['booking_bool'])
                                          ])
        print "lambdafunc created" ,datetime.now().time()
        newcols = self.df.apply(lambdafunc, axis=1)
        print "new cols created", datetime.now().time()
        newcols.columns = ['time_class', 'comp_rate_sum', 'comp_expensive', 'comp_cheap', 'target']

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

    def window_some_features(self):
        data_frames_list, targets = self.all_data()
        return data_frames_list, targets

