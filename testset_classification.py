import data_aggregator as da
import ranking as rk
import data_preprocessing as dp
import pandas as pd
import pickle as pkl

filepath_test = 'data/mini_test.csv'

data_aggregator = dp.DataPreprocessing(filepath_test)
print "adding data"
data_aggregator.add_data()
print "data added"
print "saving data"
data_aggregator.df.to_csv("data/test_set_added_variables.csv")
print "data saved"
filepath_added_variables = "data/test_set_added_variables.csv"


#can be skipped as we already have it ;)
# data_test = da.DataAggregator(filepath_added_variables)
# data_test.read_data(remove_nan=True)
data_test_df = data_aggregator.df


def make_X(testdf, select_cols):
    X = testdf.as_matrix(select_cols)
    return X


select_cols = ['prop_starrating', 'prop_review_score', 'prop_location_score2',
               'position', 'price_usd', 'promotion_flag', 'no_bookings_prop', 'no_found_prop']


<<<<<<< Updated upstream
X_test = make_X(data_test_df, select_cols)
print X_test
=======
X_test, y_test = make_X_y(data_test_df, select_cols)
print 
>>>>>>> Stashed changes

model = pkl.load(open('Classifiers_result_2\gradient_boosting_Boosting-False_max_leaf_nodes-4-learning_rate-0.1-n_estimators-100-subsample-0.5-random_state-2-min_samples_split-5-max_depth-None.pkl', 'r'))

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

rank_options = [False]

for preshuffle in rank_options:
    df_with_ranking = rk.ranking(data_test_df, y_pred, y_prob, preshuffle=preshuffle)

final_df = df_with_ranking[['srch_id', 'prop_id', 'ranking']]

final_df.to_csv('prediction_file.csv', index = False)
