import data_aggregator as da
import ranking as rk
import ndcg_calculation as ndcg
from sklearn import ensemble
from sklearn import dummy
from datetime import datetime
from sklearn.metrics import *
import pandas as pd
import numpy as np
import pickle as pkl


boost_click = True
filepathTrain = 'data\data_slice_1_added_variables.csv'
filepathTest = 'data\data_slice_2_added_variables.csv'
data = da.DataAggregator(filepathTrain)
data.read_data(remove_nan=True)
data_test = da.DataAggregator(filepathTest)
data_test.read_data(remove_nan=True)


def make_X_y(traindf, select_cols):
    y = traindf.as_matrix(['target'])[:, 0]
    X = traindf.as_matrix(select_cols)
    return X, y


select_cols = ['visitor_location_country_id', 'prop_country_id', 'prop_starrating', 'prop_review_score',
               'prop_location_score1',
               'prop_location_score2', 'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag',
               'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
               'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score',
               'orig_destination_distance']

traindf = data.df.query("click_bool == 1")
traindf = pd.concat([traindf, data.df.head(len(traindf))])
X_train_boosted, y_train_boosted = make_X_y(traindf, select_cols)
X_train_normal, y_train_normal = make_X_y(data.df, select_cols)

X_test, y_test = make_X_y(data_test.df, select_cols)

testSettings = [{'method': 'gradient_boosting',
                 'original_params': {'n_estimators': 10, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
                                     'min_samples_split': 5},
                 'param_variants': [{'learning_rate': 1.0, 'subsample': 1.0},
                                    {'learning_rate': 0.1, 'subsample': 1.0},
                                    {'learning_rate': 1.0, 'subsample': 0.5},
                                    {'learning_rate': 0.1, 'subsample': 0.5},
                                    {'learning_rate': 0.1, 'max_features': 2},
                                    {'n_estimators':100, 'learning_rate': 1.0, 'subsample': 1.0},
                                    {'n_estimators':100, 'learning_rate': 0.1, 'subsample': 1.0},
                                    {'n_estimators':100, 'learning_rate': 1.0, 'subsample': 0.5},
                                    {'n_estimators':100, 'learning_rate': 0.1, 'subsample': 0.5},
                                    {'n_estimators':100, 'learning_rate': 0.1, 'max_features': 2}]
                 },
                {'method': 'adaboost',
                 'original_params': {'n_estimators': 1000, 'learning_rate ': 1},
                 'param_variants': [{'learning_rate':0.5},
                                    {'learning_rate': 0.5},
                                    {'learning_rate': 0.1},
                                    {'n_estimators':100, 'learning_rate':0.5},
                                    {'n_estimators':100, 'learning_rate': 0.5},
                                    {'n_estimators':100, 'learning_rate': 0.1}]
                 },
                {'method': 'dummy',
                 'original_params': {},
                 'param_variants': [{}]
                 }
                ]

f = open('classif-%s.csv' % (datetime.now().strftime("%d%m%y%H%M%S")), 'w')
# f = open('classification_results.csv', 'w')
f.write('method; boosting; params; traintime; accuracy; recallmacro; recallmicro; f1macro; f1micro; meanndcg\n')
for test in testSettings:
    original_params = test['original_params']
    settings = test['param_variants']
    for setting in settings:
        print "=" * 40
        print test['method']
        params = dict(original_params)
        params.update(setting)
        print params
        clf = None
        if test['method'] == 'gradient_boosting':
            clf = ensemble.GradientBoostingClassifier(**params)
        elif test['method'] == 'adaboost':
            clf = ensemble.AdaBoostClassifier(**params)
        elif test['method'] == 'dummy':
            clf = dummy.DummyClassifier(strategy='most_frequent', random_state=None, constant=None)

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
            pkl_file = open('%s_Boosting-%s_%s.pkl' % (
                test['method'], str(boosting), "-".join([str(k) + "-" + str(v) for (k, v) in params.items()])), 'wb')
            pkl.dump(clf, pkl_file)
            traintime = datetime.now() - start_time
            print "\ttrained in", traintime
            print "\tusing settings:", params
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            print "\tclasses found", np.unique(y_pred)
            accuracy = clf.score(X_test, y_test)
            print "\taccuracy:", accuracy
            recallmacro = recall_score(y_test, y_pred, average='macro')
            print "\trecall macro:", recallmacro
            recallmicro = recall_score(y_test, y_pred, average='micro')
            print "\trecall micro:", recallmicro
            f1macro = f1_score(y_test, y_pred, average='macro')
            print "\tf1 macro:", f1macro
            f1micro = f1_score(y_test, y_pred, average='micro')
            print "\tf1 micro:", f1micro

            df_with_ranking = rk.ranking(data_test.df, y_pred, y_prob)

            search_ids = df_with_ranking['srch_id']
            diff_search_ids = search_ids.drop_duplicates()

            k = 0
            ndcg_list = []

            for id in diff_search_ids:
                mask = (df_with_ranking['srch_id'] == id)
                result_df = df_with_ranking.loc[mask]
                ndcg_result = ndcg.ndcg(result_df)
                ndcg_list.append(ndcg_result)

            meanndcg = sum(ndcg_list) / float(len(ndcg_list))
            f.write('%s; %s; %s; %s; %d; %d; %d; %d; %d; %d\n' % (
                test['method'], str(boosting), str(params), str(traintime), accuracy, recallmacro, recallmicro, f1macro,
                f1micro, meanndcg))

            print "mean ndcg", meanndcg
