import data_aggregator as da
import ranking as rk
import pandas as pd
import pickle as pkl

filepath_test = 'data/test_file1.csv'

data_test = da.DataAggregator(filepath_test)
data_test.read_data(remove_nan=True)
data_test_df = data_test.df

def make_X_y(testdf, select_cols):
    y = testdf.as_matrix(['target'])[:, 0]
    X = testdf.as_matrix(select_cols)
    return X, y


select_cols = ['prop_starrating', 'prop_review_score', 'prop_location_score2',
               'position', 'price_usd', 'promotion_flag', 'no_bookings_prop', 'no_found_prop']


X_test, y_test = make_X_y(data_test_df, select_cols)

model = pkl.load(open('Classifiers_result_1\gradient_boosting_Boosting-False_max_leaf_nodes-4-learning_rate-0.1-n_estimators-100-subsample-0.5-random_state-2-min_samples_split-5-max_depth-None.pkl', 'r'))

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

rank_options = [False]

for preshuffle in rank_options:
    df_with_ranking = rk.ranking(data_test, y_pred, y_prob, preshuffle=preshuffle)

final_df = df_with_ranking['srch_id', 'prop_id', 'ranking']

final_df.to_csv('prediction_file.csv')
