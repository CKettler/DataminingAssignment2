import data_aggregator as da
import numpy as np
import ranking as rk
import data_preprocessing as dp
import pandas as pd
import pickle as pkl
from collections import defaultdict
import math

print "open pickls"
no_bookings_dict = pkl.load(open('data/no_bookings.pkl', 'r'))
no_bookings_dict = defaultdict(int,no_bookings_dict)
print no_bookings_dict['85645529484534242173hfgcfxhsjif']
print "open pickle 1"
no_found_dict = pkl.load(open('data/no_found.pkl', 'r'))
no_found_dict = defaultdict(int,no_found_dict)

print "open pickle 2"

slices_to_do = range(8, 16)

def add_new_features(data_test_df):
    print " start"
    ones_array = np.ones(len(data_test_df))
    print " adding ones arrays"
    data_test_df['no_found_prop'] = np.transpose(ones_array)
    print " added array 1"
    data_test_df['no_bookings_prop'] = np.transpose(ones_array)
    print " added array 2"

    prop_ids = data_test_df['prop_id'].drop_duplicates()
    l = len(prop_ids)
    p = 0
    for i, id in enumerate(prop_ids):
        r = (i/float(l))*100
        if r > p:
            print r
            p = math.floor(r) + 1
        id_string = str(id)
        data_test_df.loc[data_test_df['prop_id'] == id, 'no_found_prop'] = no_found_dict[id_string]
        data_test_df.loc[data_test_df['prop_id'] == id, 'no_bookings_prop'] = no_bookings_dict[id_string]
    return data_test_df


for i in slices_to_do:
    filepath_test = 'data/test_data_slice_%d.csv' % (i)
    print "opening", filepath_test
    data_aggregator = dp.DataPreprocessing(filepath_test)
    print "adding data"
    # data_aggregator.add_data()
    data_test_df = add_new_features(data_aggregator.df)
    print "data added"
    print "saving data"
    data_aggregator.df.to_csv("data/test_set_added_variables_%i.csv" % (i))
    print "data saved"

print "all slices added"

def make_X(testdf, select_cols):
    testdf.fillna(0)
    X = testdf.as_matrix(select_cols)
    X = np.nan_to_num(X)
    return X

print "start classfieing"

model = pkl.load(open(
    'Classifiers_result_2\gradient_boosting_Boosting-False_max_leaf_nodes-4-learning_rate-0.1-n_estimators-100-subsample-0.5-random_state-2-min_samples_split-5-max_depth-None.pkl',
    'r'))

select_cols = ['prop_starrating', 'prop_review_score', 'prop_location_score2', 'price_usd',
               'promotion_flag', 'no_bookings_prop', 'no_found_prop']
rank_options = [False]

for i in slices_to_do:
    data_file = "data/test_set_added_variables_%i.csv" % (i)
    data_test_slice = dp.DataAggregator(data_file)
    data_test_df = data_test_slice.df

    X_test = make_X(data_test_df, select_cols)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    for preshuffle in rank_options:
        df_with_ranking = rk.ranking(data_test_df, y_pred, y_prob, preshuffle=preshuffle, target=False)

    final_df = df_with_ranking[['srch_id', 'prop_id']]

    final_df.to_csv('prediction_file%d.csv' % (i), index=False)
    print "slice %d done" % (i)

