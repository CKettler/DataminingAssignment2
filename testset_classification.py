import data_aggregator as da
import numpy as np
import ranking as rk
import data_preprocessing as dp
import pandas as pd
import pickle as pkl

def add_new_features(data_test_df):
    no_bookings_dict = pkl.load(open('data/no_bookings.pkl','r'))
    no_found_dict = pkl.load(open('data/no_found.pkl','r'))
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




filepath_test = 'data/mini_test.csv'

data_aggregator = dp.DataPreprocessing(filepath_test)
print "adding data"
# data_aggregator.add_data()
data_test_df = data_aggregator.df
data_test_df = add_new_features(data_test_df)
print "data added"
print "saving data"
data_aggregator.df.to_csv("data/test_set_added_variables.csv")
print "data saved"




def make_X(testdf, select_cols):
    testdf.fillna(0)
    X = testdf.as_matrix(select_cols)
    X = np.nan_to_num(X)
    return X


select_cols = ['prop_starrating', 'prop_review_score', 'prop_location_score2', 'position', 'price_usd', 'promotion_flag', 'no_bookings_prop', 'no_found_prop']


X_test = make_X(data_test_df, select_cols)


model = pkl.load(open('Classifiers_result_2\gradient_boosting_Boosting-False_max_leaf_nodes-4-learning_rate-0.1-n_estimators-100-subsample-0.5-random_state-2-min_samples_split-5-max_depth-None.pkl', 'r'))

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

rank_options = [False]

for preshuffle in rank_options:
    df_with_ranking = rk.ranking(data_test_df, y_pred, y_prob, preshuffle=preshuffle, target = False)

final_df = df_with_ranking[['srch_id', 'prop_id', 'ranking']]

final_df.to_csv('prediction_file.csv', index = False)
