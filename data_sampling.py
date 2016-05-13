import numpy as np
import random


def sampling(df, targets):
    # genereer wat random nummers zonder duplicates
    # als nummer een sessie nummer, gebruik deze sessie in de data
    # als nummer niet een sessie en dataset nog te klein (number of rows < 5000) genereer een nieuw random getal
    # else stop en return de dataset
    # met die nummers, pak een aantal sessies uit de data
    #wordt 365000
    random_integers = random.sample(xrange(1, 365), 200)

    k = 0
    for rand in random_integers:
        if rand in df['srch_id']:
            mask = (df['srch_id'] == rand)
            tf = df.loc[mask]
            if k != 0:
                # Nog even naar kijken, lijkt niet meer op dataframe
                new_df = np.vstack((new_df, tf))
                rows, cols = new_df.shape
                print new_df
                if rows > 5000:
                    break
            else:
                new_df = tf

        else:
            new_random_integer = random.sample(xrange(1,365),1)
            random_integers = np.append(random_integers, new_random_integer)


        k += 1

    print new_df

    return new_df, targets