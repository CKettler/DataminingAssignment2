import data_aggregator as da
import ranking as rk
import ndcg_calculation as ndcg
from sklearn import ensemble
from datetime import datetime
from sklearn.metrics import *
import pandas as pd
import numpy as np

boost_click = True
filepathTrain = 'data\data_slice_1_added_variables.csv'
filepathTest = 'data\data_slice_2_added_variables.csv'
data = da.DataAggregator(filepathTrain)
data.read_data(remove_nan=True)


def make_X_y(traindf, select_cols):
    y_train = traindf.as_matrix(['target'])[:, 0]
    X_train = traindf.as_matrix(select_cols)
    return X_train, y_train


select_cols = ['visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
               'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1',
               'prop_location_score2', 'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag',
               'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
               'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score',
               'orig_destination_distance']

traindf = data.df.query("click_bool == 1")
traindf = pd.concat([traindf, data.df.head(len(traindf))])
X_train_boosted, y_train_boosted = make_X_y(traindf, select_cols)
X_train_normal, y_train_normal = make_X_y(data.df, select_cols)

testSettings = [{'method': 'gradient_boosting',
                 'original_params': {'n_estimators': 10, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
                                     'min_samples_split': 5},
                 'param_variants': [{'learning_rate': 1.0, 'subsample': 1.0},
                                    {'learning_rate': 0.1, 'subsample': 1.0},
                                    {'learning_rate': 1.0, 'subsample': 0.5},
                                    {'learning_rate': 0.1, 'subsample': 0.5},
                                    {'learning_rate': 0.1, 'max_features': 2}]
                 }]

for test in testSettings:
    original_params = test['original_params']
    settings = test['param_variants']
    for setting in settings:
        print "="*40
        print test['method']
        params = dict(original_params)
        params.update(setting)
        print params
        clf = None
        if test['method'] == 'gradient_boosting':
            clf = ensemble.GradientBoostingClassifier(**params)
        for boosting in [True, False]:
            print "Boosting", boosting
            if boosting:
                X_train = X_train_boosted
                y_train = y_train_boosted
                df = traindf
            else:
                X_train = X_train_normal
                y_train = y_train_normal
                df = data.df

            start_time = datetime.now()
            clf.fit(X_train, y_train)
            print "\ttrained in", datetime.now() - start_time, "using settings:", params
            y_pred = clf.predict(X_train)
            y_prob = clf.predict_proba(X_train)
            print "\tclasses found", np.unique(y_pred)
            print "\taccuracy:", clf.score(X_train, y_train)
            print "\trecall macro:", recall_score(y_train, y_pred, average='macro')
            print "\trecall micro:", recall_score(y_train, y_pred, average='micro')
            print "\tf1 macro:", f1_score(y_train, y_pred, average='macro')
            print "\tf1 micro:", f1_score(y_train, y_pred, average='micro')

            df_with_ranking = rk.ranking(df, y_pred, y_prob)

            search_ids = df_with_ranking['srch_id']
            diff_search_ids = search_ids.drop_duplicates()

            k = 0
            ndcg_list = []

            for id in diff_search_ids:
                mask = (df_with_ranking['srch_id'] == id)
                result_df = df_with_ranking.loc[mask]
                ndcg = ndcg.ndcg(result_df, k)
                ndcg_list.append([ndcg])
