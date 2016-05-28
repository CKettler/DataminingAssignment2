import pandas as pd
import numpy as np
import data_aggregator as da
from datetime import datetime

pd.options.mode.chained_assignment = None


def ranking(df, y_pred, y_prob, preshuffle = False, target = False):
    print "start ranking", datetime.now()
    if target:
        df = df[['date_time', 'srch_id', 'prop_id', 'target']]
    else:
        df = df[['date_time', 'srch_id', 'prop_id']]
    df['y_pred'] = y_pred
    df['y_prob_0'] = y_prob[:, 0]
    df['y_prob_1'] = y_prob[:, 1]
    df['y_prob_6'] = y_prob[:, 2]

    search_ids = df['srch_id']
    # to get all different search id's to rank for different sessions
    # problem with the new dataset with more clicked is that we loose a big part of the session. Are we going to search
    # for the whole session in the original data or the dataslice?
    diff_search_ids = search_ids.drop_duplicates()

    k = 0

    for id in diff_search_ids:
        mask = (df['srch_id'] == id)
        tf = df.loc[mask]
        if len(tf) > 1:
            if preshuffle:
                tf.iloc[np.random.permutation(len(tf))]
                result = tf.sort_values(by=['y_prob_6'], ascending=False)
            else:
                result = tf.sort_values(by=['y_prob_6'], ascending=False)
            result['ranking'] = range(1, len(result['srch_id']) + 1)
        else:
            result = tf
            result['ranking'] = 1
        if target:
            rank = result[['date_time', 'ranking', 'srch_id', 'prop_id', 'target']]
        else:
            rank = result[['date_time', 'ranking', 'srch_id', 'prop_id']]
        if k != 0:
            df_list = [new_df, rank]
            new_df = pd.concat(df_list)
        else:
            new_df = rank
        k += 1

    print "created ranked df", datetime.now()

    return new_df
