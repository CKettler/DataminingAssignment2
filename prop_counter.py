import pandas as pd
from collections import Counter
import pickle as pkl

pkl_no_found = open('no_found.pkl', 'w')
pkl_no_bookings = open('no_bookings.pkl', 'w')

filepath = 'data/training_set_VU_DM_2014.csv'
# filepath = 'data_slice_1.csv'
df = pd.read_csv(filepath)

no_found = dict(Counter(df['prop_id']))
pkl.dump(no_found, pkl_no_found)

df = df.query("click_bool == 1")

no_bookings = dict(Counter(df['prop_id']))
pkl.dump(no_bookings, pkl_no_bookings)

pkl_no_found.close()
pkl_no_bookings.close()
