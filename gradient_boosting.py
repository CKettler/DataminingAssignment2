import data_aggregator as da
import ranking as rk
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

original_params = {'n_estimators': 10, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
                   'min_samples_split': 5}

for label, color, setting in [('No shrinkage', 'orange',
                               {'learning_rate': 1.0, 'subsample': 1.0}),
                              ('learning_rate=0.1', 'turquoise',
                               {'learning_rate': 0.1, 'subsample': 1.0}),
                              ('subsample=0.5', 'blue',
                               {'learning_rate': 1.0, 'subsample': 0.5}),
                              ('learning_rate=0.1, subsample=0.5', 'gray',
                               {'learning_rate': 0.1, 'subsample': 0.5}),
                              ('learning_rate=0.1, max_features=2', 'magenta',
                               {'learning_rate': 0.1, 'max_features': 2})]:
    params = dict(original_params)
    params.update(setting)
    clf = ensemble.GradientBoostingClassifier(**params)
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
    df_with_ranking.to_csv("data/rankings_data_slice_1")