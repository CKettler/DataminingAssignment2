import data_aggregator as da
from sklearn import ensemble
from datetime import datetime
from sklearn.metrics import recall_score
import numpy as np


filepath = 'data/data_slice_1_added_variables.csv'
data = da.DataAggregator(filepath)
data.read_data()
print data.variables
y_train = data.select_variables(['booking_bool', 'click_bool', 'target']).as_matrix(['target'])[:,0]
select_cols = ['visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance']
# data.keep_df_variables(select_cols)
data.df = data.df.fillna(0)
X_train = data.df.as_matrix(select_cols)
print X_train
print y_train

original_params = {'n_estimators': 3, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
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
    print "start training"
    start_time = datetime.now()
    clf.fit(X_train, y_train)
    print clf.classes_
    print "trained in", datetime.now() - start_time

    y_pred = clf.predict(X_train)
    print y_pred
    print clf.score(X_train, y_train)
    print recall_score(y_train, y_pred, average='macro')