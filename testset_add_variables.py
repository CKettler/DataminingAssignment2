import data_aggregator as da
import numpy as np
import ranking as rk
import data_preprocessing as dp
import pandas as pd
import pickle as pkl

no_bookings_dict = pkl.load(open('data/no_bookings.pkl', 'r'))
no_found_dict = pkl.load(open('data/no_found.pkl', 'r'))


def add_new_features(data_test_df):
    prop_ids = data_test_df['prop_id']
    diff_prop_ids = prop_ids.drop_duplicates()

    k = 0

    for id in diff_prop_ids:
        id_string = str(id)
        mask = (data_test_df['prop_id'] == id)
        tf = data_test_df.loc[mask]
        ones_array = np.ones(len(tf))

        if id_string in no_found_dict:
            feature_found_array = no_found_dict[id_string] * ones_array
        else:
            feature_found_array = 0 * ones_array
        if id_string in no_bookings_dict:
            feature_book_array = no_bookings_dict[id_string] * ones_array
        else:
            feature_book_array = 0 * ones_array
        tf['no_found_prop'] = feature_found_array
        tf['no_bookings_prop'] = feature_book_array
        if k != 0:
            df_list = [new_df, tf]
            new_df = pd.concat(df_list)
        else:
            new_df = tf
        k += 1
    return new_df



for i in range(13, 16):
    filepath_test = 'data/test_data_slice_%d.csv' % (i)
    print "opening", filepath_test
    data_aggregator = dp.DataPreprocessing(filepath_test)
    print "adding data"
    # data_aggregator.add_data()
    data_test_df = data_aggregator.df
    data_test_df = add_new_features(data_test_df)
    print "data added"
    print "saving data"
    data_aggregator.df.to_csv("data/test_set_added_variables_%i.csv" % (i))
    print "data saved"

