import pandas as pd
import numpy as np
import data_aggregator as da
from datetime import datetime


def ranking(df, y_pred, y_prob):
    df = df[['srch_id','prop_id']]
    df['y_pred'] = y_pred
    print y_prob[:,0]
    df['y_prob_0'] = y_prob[:,0]
    df['y_prob_1'] = y_prob[:,1]
    df['y_prob_6'] = y_prob[:,2]
    print df

