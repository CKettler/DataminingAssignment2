import numpy as np
import random
import pandas as pd


def sampling(df, targets):
    # genereer wat random nummers zonder duplicates
    # als nummer een sessie nummer, gebruik deze sessie in de data
    # als nummer niet een sessie en dataset nog te klein (number of rows < 5000) genereer een nieuw random getal
    # else stop en return de dataset
    # met die nummers, pak een aantal sessies uit de data
    #wordt 365000
    #random_integers = random.sample(xrange(1, 365), 200)

    k = 0
    rows = 0
    random_integers = []
    while rows < 2000:
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
                # Nog even naar kijken, lijkt niet meer op dataframe
                df_list = [new_df,tf]
                new_df = pd.concat(df_list)
                print new_df
                #new_df = np.vstack((new_df, tf))
                rows, cols = new_df.shape
                # print new_df
                # if rows > 5000:
                #    break
            else:
                print 'made new df'
                new_df = tf
        else:
            continue

        k += 1

    print new_df

    return new_df, targets