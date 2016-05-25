import data_aggregator as da
import ranking as rk
import ndcg_calculation as ndcg
from sklearn import ensemble
from datetime import datetime
from sklearn.metrics import *
import pandas as pd
import numpy as np


filepath = 'C:\Users\Celeste\Documents\GitHub\DataminingAssignment2\data\data_slice_1_added_variables.csv'
data = da.DataAggregator(filepath)
data.read_data()

traindf = data.df.query("click_bool == 1")
clickedlen = len(traindf)
traindf = pd.concat([traindf, data.df.head(clickedlen)])

print data.variables
y_train = traindf.as_matrix(['target'])[:,0]
select_cols = ['visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance']
# data.keep_df_variables(select_cols)
traindf = traindf.fillna(0)
X_train = traindf.as_matrix(select_cols)
print X_train
print y_train

original_params = {'max_leaf_nodes': None, 'max_depth': None, 'random_state': None,
                   'min_samples_split': 2}

for label, color, setting in [('3 trees', 'orange',
                               {'n_estimators': 3}),
                              ('10 trees', 'turquoise',
                               {'n_estimators': 10}),
                              ('30 trees', 'blue',
                               {'n_estimators': 30})]:
    params = dict(original_params)
    params.update(setting)
    clf = ensemble.RandomForestClassifier(**params)
    start_time = datetime.now()
    clf.fit(X_train, y_train)
    print clf.classes_
    print "trained in", datetime.now() - start_time

    y_pred = clf.predict(X_train)
    y_prob = clf.predict_proba(X_train)
    print "class probs", y_prob
    print "classes found", np.unique(y_pred)
    print "accuracy:", clf.score(X_train, y_train)
    print "recall macro:", recall_score(y_train, y_pred, average='macro')
    print "recall micro:", recall_score(y_train, y_pred, average='micro')
    print "f1 macro:", f1_score(y_train, y_pred, average='macro')
    print "f1 micro:", f1_score(y_train, y_pred, average='micro')

    df_with_ranking = rk.ranking(traindf, y_pred, y_prob)

    search_ids = df_with_ranking['srch_id']
    diff_search_ids = search_ids.drop_duplicates()

    k = 0
    ndcg_list = []

    for id in diff_search_ids:
        mask = (df_with_ranking['srch_id'] == id)
        result_df = df_with_ranking.loc[mask]
        ndcg = ndcg.ndcg(result_df, k)
        ndcg_list.append([ndcg])

