import numpy as np
import random
import pandas as pd


def sampling(df):
    k = 0
    rows = 0
    random_integers = []
    while rows < 2000:
        # 365 wordt 365000
        new_random_integer = random.sample(xrange(1, 365), 1)
        while new_random_integer in random_integers:
            new_random_integer = random.sample(xrange(1, 365), 1)
            print new_random_integer
        random_integers = np.append(random_integers, new_random_integer)
        print random_integers
        if new_random_integer[len(new_random_integer)-1] in df['srch_id']:
            mask = (df['srch_id'] == new_random_integer)
            tf = df.loc[mask]
            if k != 0:
                df_list = [new_df,tf]
                new_df = pd.concat(df_list)
                print new_df
                rows, cols = new_df.shape
            else:
                print 'made new df'
                new_df = tf
        else:
            continue

        k += 1

    print new_df

    return new_df