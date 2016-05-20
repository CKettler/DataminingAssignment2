import data_aggregator as da
import numpy as np
from sklearn import dummy
from datetime import datetime


def classification(data_matrix, target_matrix, test_matrix, strategy = 'most_frequent'):
    print "data detected", datetime.now().time()
    model = dummy.DummyClassifier(strategy=strategy, random_state=None, constant=None)
    print "model made", datetime.now().time()
    model.fit(data_matrix, target_matrix)
    print "model fitted", datetime.now().time()
    results = model.predict(test_matrix)
    print results




filepath = 'C:\Users\Celeste\Documents\GitHub\DataminingAssignment2\data\data_slice_1_added_variables.csv'

data_aggregator = da.DataAggregator(filepath)
data_aggregator.read_data()
df = data_aggregator.df
variables = list(df.columns.values)
target = data_aggregator.df['target'].as_matrix()
data_aggregator.keep_df_variables(['srch_id', 'date_time', 'site_id', 'visitor_location_country_id',
                                        'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
                                        'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
                                        'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price',
                                        'position', 'price_usd', 'promotion_flag', 'srch_destination_id',
                                        'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
                                        'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
                                        'srch_query_affinity_score', 'orig_destination_distance', 'random_bool',
                                        'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate',
                                        'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv',
                                        'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff',
                                        'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv',
                                        'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff',
                                        'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'click_bool',
                                        'gross_bookings_usd', 'booking_bool', 'no_bookings_prop', 'no_found_prop',
                                        'time_class', 'comp_rate_sum', 'comp_expensive', 'comp_cheap', 'target'])
data_matrix = data_aggregator.df.as_matrix()

test_matrix = data_matrix

classification(data_matrix, target, test_matrix, strategy = 'most_frequent')