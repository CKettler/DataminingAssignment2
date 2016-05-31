import numpy as np
import ranking as rk
import data_preprocessing as dp
import pickle as pkl


def make_X(testdf, select_cols):
    testdf.fillna(0)
    X = testdf.as_matrix(select_cols)
    X = np.nan_to_num(X)
    return X


for i in range(13, 16):
    data_file = "data/test_set_added_variables_%i.csv" % (i)
    data_test_slice = dp.DataAggregator(data_file)
    data_test_df = data_test_slice.df

    select_cols = ['prop_starrating', 'prop_review_score', 'prop_location_score2', 'position', 'price_usd',
                   'promotion_flag', 'no_bookings_prop', 'no_found_prop']

    X_test = make_X(data_test_df, select_cols)

    model = pkl.load(open(
        'Classifiers_result_2\gradient_boosting_Boosting-False_max_leaf_nodes-4-learning_rate-0.1-n_estimators-100-subsample-0.5-random_state-2-min_samples_split-5-max_depth-None.pkl',
        'r'))

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    rank_options = [False]

    for preshuffle in rank_options:
        df_with_ranking = rk.ranking(data_test_df, y_pred, y_prob, preshuffle=preshuffle, target=False)

    final_df = df_with_ranking[['srch_id', 'prop_id']]

    final_df.to_csv('prediction_file%d.csv' % (i), index=False)
