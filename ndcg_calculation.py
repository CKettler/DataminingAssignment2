import pandas as pd
import numpy as np
import math
import data_aggregator as da
from datetime import datetime

#The DCG measure calculation
#Here the list with all documents containing the words of the query is inserted as results list
#It is sorted, to create the list for DCGi
#Then both DCG and DCGi are calculated for all documents
#DCG is divided by DCGi which results in nDCG
def ndcg(result_df,k):
    dcg_list = []
    dcgi_list = []
    # mask = (result_df['target'] == 6)
    # booked_df = result_df.loc[mask]
    # booked_df['discounted_score'] = 5 * np.ones(len(booked_df))
    # mask = (result_df['target'] == 1)
    # clicked_df = result_df.loc[mask]
    # clicked_df['discounted_score'] = np.ones(len(booked_df))
    # sorted_result_list = pd.concat([booked_df, clicked_df])
    # mask = (result_df['target'] == 0)
    # rest_df = result_df.loc[mask]
    # rest_df['discounted_score'] = np.ones(len(booked_df))
    # sorted_result_list = pd.concat([rest_df, sorted_result_list])
    sorted_result_df = result_df.sort_values(by=['target'], ascending=False)
    print sorted_result_df
    #print sorted_result_list
    for i in range(len(result_df)):
        print result_df['target'][i]
        relevance_rank = result_df['target'][i]
        if i > k:
            break
        dcg_rank = ((2**(relevance_rank))-1)/(math.log(1+(1+i),2))
        dcg_list.append(dcg_rank)
    for j in range(len(sorted_result_df)):
        relevance_i_rank = result_df['target'][j]
        if j > k:
            break
        dcgi_rank = ((2**relevance_i_rank)-1)/(math.log(1+(1+j),2))
        dcgi_list.append(dcgi_rank)
    dcgp = sum(dcg_list)
    dcgi = sum(dcgi_list)
    ndcg = dcgp/dcgi
    print ndcg
    return ndcg
