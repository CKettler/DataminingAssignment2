import pandas as pd
from datetime import datetime
import numpy as np
import math


class DataAggregator:
    def __init__(self, filepath, window_size=5, variables=None):
        self.filepath = filepath
        self.df = pd.DataFrame()
        self.num_rows = 0
        self.site_ids = []
        self.variables = []

    def read_data(self, remove_nan=False):
        start_time = datetime.now()
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        self.df = pd.read_csv(self.filepath, parse_dates=['date_time'], date_parser=dateparse)
        self.num_rows = len(self.df)
        print "num rows:", self.num_rows
        self.df[['comp%d_rate' % i for i in range(1, 9)]] = self.df[['comp%d_rate' % i for i in range(1, 9)]].fillna(axis=1, method='backfill')
        self.site_ids = self.df['site_id'].unique()  # all participants (don't know if needed)
        self.variables = list(self.df.columns.values)  # all possible variable names
        # print self.variables
        if remove_nan:
            self.df = self.df.fillna(0)
        print "data opened in ", datetime.now() - start_time

    def keep_df_variables(self, variables):
        self.df = self.df[variables]

    def select_variables(self, variables):
        return self.df[variables]