import pandas as pd
import numpy as np
import data_aggregator as da
from datetime import datetime


def ranking(df, y_pred, y_prob):
    df = df[['srch_id','prop_id']]
    df['y_pred'] = y_pred
    df['y_prob_0'] = y_prob[:,0]
    df['y_prob_1'] = y_prob[:,1]
    df['y_prob_6'] = y_prob[:,2]
    print "added collumns", datetime.now()

    search_ids = df['srch_id']
    # to get all different search id's to rank for different sessions
    # problem with the new dataset with more clicked is that we loose a big part of the session. Are we going to search
    # for the whole session in the original data or the dataslice?
    diff_search_ids = search_ids.drop_duplicates()

    k = 0

    for id in diff_search_ids:
        mask = (df['srch_id'] == id )
        tf = df.loc[mask]
        if len(tf) > 1:
            result = tf.sort_values(by=['y_prob_6'], ascending = False)
            result['ranking'] = range(1,len(result['srch_id'])+1)
        else:
            result = tf
            result['ranking'] = 1
        rank = result[['ranking','srch_id','prop_id']]
        if k != 0:
            df_list = [new_df, rank]
            new_df = pd.concat(df_list)
        else:
            new_df = rank
        k += 1

    print "created ranked df", datetime.now()

    return new_df